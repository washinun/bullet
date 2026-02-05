mod direct;
mod montybinpack;
mod rng;
mod sfbinpack;
mod text;
pub mod viribinpack;

pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::ViriBinpackLoader;

use acyclib::device::tensor::Shape;
use bulletformat::BulletFormat;

use crate::game::{inputs::SparseInputType, outputs::OutputBuckets};

use super::Wgt;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, start_batch: usize, batch_size: usize, f: F);
}

pub(crate) type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;

#[derive(Clone)]
pub struct DefaultDataLoader<I: SparseInputType, O, D> {
    input_getter: I,
    output_getter: O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    scale: f32,
    loader: D,
}

impl<I: SparseInputType, O, D> DefaultDataLoader<I, O, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        scale: f32,
        loader: D,
    ) -> Self {
        Self { input_getter, output_getter, blend_getter, weight_getter, use_win_rate_model, wdl, scale, loader }
    }
}

impl<I, O, D> DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    pub fn load_and_map_batches<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        f: F,
    ) {
        self.loader.map_batches(start_batch, batch_size, f);
    }

    pub fn prepare(&self, data: &[I::RequiredDataType], threads: usize, blend: f32) -> PreparedData<I, O> {
        PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
        )
    }

    /// Prepare data for NNUE loss (separate score and outcome targets with per-sample WDL)
    pub fn prepare_nnue(&self, data: &[I::RequiredDataType], threads: usize, wdl: f32) -> PreparedData<I, O> {
        PreparedData::new_nnue(
            self.input_getter.clone(),
            self.output_getter,
            self.weight_getter,
            data,
            threads,
            self.scale,
            wdl,
        )
    }
}

pub(crate) struct DenseInput {
    pub value: Vec<f32>,
    pub shape: Shape,
}

#[derive(Clone)]
pub(crate) struct SparseInput {
    pub value: Vec<i32>,
    pub max_active: usize,
    pub shape: Shape,
}

/// A batch of data, in the correct format for the GPU.
pub struct PreparedData<I: SparseInputType, O> {
    pub(crate) input_getter: I,
    pub(crate) output_getter: O,
    pub(crate) batch_size: usize,
    pub(crate) stm: SparseInput,
    pub(crate) nstm: SparseInput,
    pub(crate) buckets: SparseInput,
    pub(crate) targets: DenseInput,
    pub(crate) weights: DenseInput,
    /// NNUE loss targets: 3-channel [score_winrate, outcome, wdl]
    pub(crate) nnue_targets: Option<DenseInput>,
}

impl<I, O> PreparedData<I, O>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);
        let input_size = input_getter.num_inputs();
        let output_size = if wdl { 3 } else { 1 };
        let sparse_size = max_active * batch_size;

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: SparseInput { max_active, value: vec![0; sparse_size], shape: Shape::new(input_size, 1) },
            nstm: SparseInput { max_active, value: vec![0; sparse_size], shape: Shape::new(input_size, 1) },
            buckets: SparseInput { max_active: 1, value: vec![0; batch_size], shape: Shape::new(O::BUCKETS, 1) },
            targets: DenseInput { value: vec![0.0; output_size * batch_size], shape: Shape::new(output_size, 1) },
            weights: DenseInput { value: vec![0.0; batch_size], shape: Shape::new(1, 1) },
            nnue_targets: None,
        };

        let sparse_chunk_size = max_active * chunk_size;

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(sparse_chunk_size))
                .zip(prep.nstm.value.chunks_mut(sparse_chunk_size))
                .zip(prep.buckets.value.chunks_mut(chunk_size))
                .zip(prep.targets.value.chunks_mut(output_size * chunk_size))
                .zip(prep.weights.value.chunks_mut(chunk_size))
                .for_each(
                    |(((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk)| {
                        let inp = &prep.input_getter;
                        let out = &prep.output_getter;
                        s.spawn(move || {
                            let chunk_len = data_chunk.len();

                            for i in 0..chunk_len {
                                let pos = &data_chunk[i];
                                let mut j = 0;
                                let sparse_offset = max_active * i;

                                inp.map_features(pos, |our, opp| {
                                    assert!(
                                        our < input_size && opp < input_size,
                                        "Input feature index exceeded input size!"
                                    );

                                    stm_chunk[sparse_offset + j] = our as i32;
                                    nstm_chunk[sparse_offset + j] = opp as i32;

                                    j += 1;
                                });

                                for j in j..max_active {
                                    stm_chunk[sparse_offset + j] = -1;
                                    nstm_chunk[sparse_offset + j] = -1;
                                }

                                assert!(j <= max_active, "More inputs provided than the specified maximum!");

                                buckets_chunk[i] = i32::from(out.bucket(pos));
                                weights_chunk[i] = weight_getter.map_or(1.0, |w| w(pos));

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = f32::from(pos.score());
                                    let score = if use_win_rate_model {
                                        let p = (score - 270.0) / 380.0;
                                        let pm = (-score - 270.0) / 380.0;
                                        0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
                                    } else {
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }
                        });
                    },
                );
        });

        prep
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

/// YaneuraOu-style win rate model
/// win_rate = 0.5 * (1.0 + sigmoid((score - 270.0) / 380.0) - sigmoid((-score - 270.0) / 380.0))
pub fn yaneuraou_win_rate(score: f32) -> f32 {
    let p = (score - 270.0) / 380.0;
    let pm = (-score - 270.0) / 380.0;
    0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
}

impl<I, O> PreparedData<I, O>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    /// Creates empty sparse inputs for STM and NSTM perspectives.
    fn create_empty_sparse_inputs(
        batch_size: usize,
        max_active: usize,
        input_size: usize,
    ) -> (SparseInput, SparseInput) {
        let sparse_size = max_active * batch_size;
        (
            SparseInput { max_active, value: vec![0; sparse_size], shape: Shape::new(input_size, 1) },
            SparseInput { max_active, value: vec![0; sparse_size], shape: Shape::new(input_size, 1) },
        )
    }

    /// Creates empty dense inputs for targets and weights (3-channel: [score, outcome, wdl]).
    fn create_empty_dense_inputs(batch_size: usize) -> (DenseInput, DenseInput, DenseInput) {
        (
            // Compatibility target (not used for NNUE loss)
            DenseInput { value: vec![0.0; batch_size], shape: Shape::new(1, 1) },
            // Weights
            DenseInput { value: vec![0.0; batch_size], shape: Shape::new(1, 1) },
            // 3-channel target: [score_winrate, outcome, wdl]
            DenseInput { value: vec![0.0; batch_size * 3], shape: Shape::new(3, 1) },
        )
    }

    /// Prepare data for NNUE loss with separate score and outcome targets.
    ///
    /// This method processes the input data in parallel, extracting:
    /// - Sparse features for both perspectives (STM and NSTM)
    /// - 3-channel targets: [score_winrate, outcome, wdl]
    ///   - score_winrate: Win rate from evaluation using YaneuraOu model
    ///   - outcome: Game results (0.0=loss, 0.5=draw, 1.0=win)
    ///   - wdl: Per-sample WDL value for blending (0.0=score only, 1.0=outcome only)
    /// - Position weights
    pub fn new_nnue(
        input_getter: I,
        output_getter: O,
        weight_getter: Option<Wgt<I>>,
        data: &[I::RequiredDataType],
        threads: usize,
        scale: f32,
        wdl: f32,
    ) -> Self {
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);
        let input_size = input_getter.num_inputs();

        let (stm, nstm) = Self::create_empty_sparse_inputs(batch_size, max_active, input_size);
        let (targets, weights, targets_3ch) = Self::create_empty_dense_inputs(batch_size);

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm,
            nstm,
            buckets: SparseInput { max_active: 1, value: vec![0; batch_size], shape: Shape::new(O::BUCKETS, 1) },
            targets,
            weights,
            nnue_targets: Some(targets_3ch),
        };

        prep.process_data_in_parallel(data, chunk_size, max_active, input_size, scale, weight_getter, wdl);
        prep
    }

    /// Processes data chunks in parallel threads.
    fn process_data_in_parallel(
        &mut self,
        data: &[I::RequiredDataType],
        chunk_size: usize,
        max_active: usize,
        input_size: usize,
        scale: f32,
        weight_getter: Option<Wgt<I>>,
        wdl: f32,
    ) {
        let sparse_chunk_size = max_active * chunk_size;
        let inp = &self.input_getter;
        let out = &self.output_getter;

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(self.stm.value.chunks_mut(sparse_chunk_size))
                .zip(self.nstm.value.chunks_mut(sparse_chunk_size))
                .zip(self.buckets.value.chunks_mut(chunk_size))
                .zip(self.nnue_targets.as_mut().unwrap().value.chunks_mut(chunk_size * 3))
                .zip(self.weights.value.chunks_mut(chunk_size))
                .for_each(
                    |(((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), targets_chunk), weights_chunk)| {
                        Self::process_chunk_internal(
                            s,
                            data_chunk,
                            stm_chunk,
                            nstm_chunk,
                            buckets_chunk,
                            targets_chunk,
                            weights_chunk,
                            max_active,
                            input_size,
                            scale,
                            weight_getter,
                            wdl,
                            inp,
                            out,
                        );
                    },
                );
        });
    }

    /// Processes a single chunk of data in a separate thread (static method).
    fn process_chunk_internal<'scope>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        data_chunk: &'scope [I::RequiredDataType],
        stm_chunk: &'scope mut [i32],
        nstm_chunk: &'scope mut [i32],
        buckets_chunk: &'scope mut [i32],
        targets_chunk: &'scope mut [f32],
        weights_chunk: &'scope mut [f32],
        max_active: usize,
        input_size: usize,
        scale: f32,
        weight_getter: Option<Wgt<I>>,
        wdl: f32,
        inp: &'scope I,
        out: &'scope O,
    ) {
        scope.spawn(move || {
            for (i, pos) in data_chunk.iter().enumerate() {
                Self::process_single_position(
                    pos,
                    i,
                    stm_chunk,
                    nstm_chunk,
                    buckets_chunk,
                    targets_chunk,
                    weights_chunk,
                    max_active,
                    input_size,
                    scale,
                    weight_getter,
                    wdl,
                    inp,
                    out,
                );
            }
        });
    }

    /// Processes a single position, extracting features and targets.
    #[allow(clippy::too_many_arguments)]
    fn process_single_position<'a>(
        pos: &'a I::RequiredDataType,
        idx: usize,
        stm_chunk: &'a mut [i32],
        nstm_chunk: &'a mut [i32],
        buckets_chunk: &'a mut [i32],
        targets_chunk: &'a mut [f32],
        weights_chunk: &'a mut [f32],
        max_active: usize,
        input_size: usize,
        scale: f32,
        weight_getter: Option<Wgt<I>>,
        wdl: f32,
        inp: &'a I,
        out: &'a O,
    ) {
        let sparse_offset = max_active * idx;
        let mut feature_count = 0;

        // Extract sparse input features
        inp.map_features(pos, |our, opp| {
            assert!(
                our < input_size && opp < input_size,
                "Feature index {} or {} exceeds input size {}",
                our,
                opp,
                input_size
            );
            stm_chunk[sparse_offset + feature_count] = our as i32;
            nstm_chunk[sparse_offset + feature_count] = opp as i32;
            feature_count += 1;
        });

        // Pad remaining slots with -1
        for j in feature_count..max_active {
            stm_chunk[sparse_offset + j] = -1;
            nstm_chunk[sparse_offset + j] = -1;
        }

        assert!(
            feature_count <= max_active,
            "Feature count {} exceeds maximum {} for position {}",
            feature_count,
            max_active,
            idx
        );

        // Set output bucket
        buckets_chunk[idx] = i32::from(out.bucket(pos));

        // Set position weight
        weights_chunk[idx] = weight_getter.map_or(1.0, |w| w(pos));

        // Extract and convert score to win rate
        let score = f32::from(pos.score());
        let score_winrate = yaneuraou_win_rate(score / scale);

        // Extract game outcome
        let outcome = f32::from(pos.result() as u8) / 2.0;

        // Store 3-channel target: [score_winrate, outcome, wdl]
        let target_offset = idx * 3;
        targets_chunk[target_offset] = score_winrate;
        targets_chunk[target_offset + 1] = outcome;
        targets_chunk[target_offset + 2] = wdl;
    }
}

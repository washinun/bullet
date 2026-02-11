//! Shogi Data Loader with fen-skipping support
//!
//! Specialized data loader for PackedSfenValue that supports:
//! - random-fen-skipping: Skip positions randomly with configurable probability
//! - early-fen-skipping: Skip positions based on ply (move count)
//!
//! When skipping is enabled, the loader buffers filtered positions to maintain
//! the target batch size, reading additional data as needed.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};

use super::packed_sfen::PackedSfenValue;
use crate::value::loader::{DataLoader, SimpleRand};

/// Maximum number of iterations to prevent infinite loops when filtering.
/// If we can't collect enough data after this many passes through the files,
/// we'll emit what we have and continue.
const MAX_ITERATIONS: usize = 10;

/// Specialized data loader for Shogi (PackedSfenValue) with fen-skipping support
#[derive(Clone)]
pub struct ShogiDirectSequentialDataLoader {
    file_paths: Vec<String>,
    random_fen_skipping: u32,
    early_fen_skipping: u32,
}

impl ShogiDirectSequentialDataLoader {
    pub fn new(file_paths: &[&str]) -> Self {
        let file_paths = file_paths.iter().map(|path| path.to_string()).collect::<Vec<_>>();

        for path in &file_paths {
            let path_buf: PathBuf = path.parse().unwrap();
            assert!(path_buf.exists(), "File not found: {path}");
        }

        Self { file_paths, random_fen_skipping: 0, early_fen_skipping: 0 }
    }

    /// Set random FEN skipping.
    /// `n` means on average skip `n` positions before using one.
    /// For example, n=3 means use 1 out of every 4 positions (1/(n+1) probability).
    pub fn with_random_fen_skipping(mut self, n: u32) -> Self {
        self.random_fen_skipping = n;
        self
    }

    /// Set early FEN skipping based on ply (move count).
    /// Positions with ply < `n` will be skipped.
    pub fn with_early_fen_skipping(mut self, n: u32) -> Self {
        self.early_fen_skipping = n;
        self
    }

    fn map_file_sizes<F: FnMut(&str, u64)>(&self, mut f: F) {
        for file in self.file_paths.iter() {
            f(file, std::fs::metadata(file).unwrap().len());
        }
    }

    /// Emit full batches from the pending buffer.
    /// Returns true if the callback requested to break, false otherwise.
    fn emit_batches<F: FnMut(&[PackedSfenValue]) -> bool>(
        pending: &mut Vec<PackedSfenValue>,
        batch_size: usize,
        f: &mut F,
    ) -> bool {
        while pending.len() >= batch_size {
            let batch_to_emit: Vec<PackedSfenValue> = pending.drain(0..batch_size).collect();
            if f(&batch_to_emit) {
                return true;
            }
        }
        false
    }
}

impl DataLoader<PackedSfenValue> for ShogiDirectSequentialDataLoader {
    fn data_file_paths(&self) -> &[String] {
        &self.file_paths
    }

    fn count_positions(&self) -> Option<u64> {
        let data_size = std::mem::size_of::<PackedSfenValue>() as u64;

        let mut file_size = 0;

        self.map_file_sizes(|file, this_size| {
            if this_size % data_size != 0 {
                panic!("File [{file}] does not have a multiple of {data_size} size!");
            }

            file_size += this_size;
        });

        Some(file_size / data_size)
    }

    fn map_batches<F: FnMut(&[PackedSfenValue]) -> bool>(&self, start_batch: usize, batch_size: usize, mut f: F) {
        let buffer_size_mb = 256;
        let buffer_size = buffer_size_mb * 1024 * 1024;
        let data_size = std::mem::size_of::<PackedSfenValue>();
        let batches_per_load = buffer_size / data_size / batch_size;
        let cap = batch_size * batches_per_load;

        let data_size_u64 = data_size as u64;

        let mut batches_per_epoch = 0;
        self.map_file_sizes(|_, this_size| {
            batches_per_epoch += (this_size / data_size_u64).div_ceil(batch_size as u64)
        });

        let start_point = start_batch % batches_per_epoch as usize;

        let mut start_file_idx = 0;
        let mut net_batches = 0;
        for file in self.file_paths.iter() {
            let this_size = std::fs::metadata(file).unwrap().len();
            let this_batches = (this_size / data_size_u64).div_ceil(batch_size as u64);

            net_batches += this_batches;

            if start_point < net_batches as usize {
                net_batches -= this_batches;
                break;
            } else {
                start_file_idx += 1;
            }
        }

        let mut file_paths = self.file_paths.clone();
        file_paths.rotate_left(start_file_idx);

        let mut to_skip = (start_point - net_batches as usize) * batch_size;

        // Initialize read buffer
        let mut read_buf: Vec<PackedSfenValue> = Vec::with_capacity(cap);
        unsafe {
            read_buf.set_len(cap);
        }

        // Initialize pending buffer for filtered positions
        // This buffer accumulates filtered positions until we have enough for a batch
        let mut pending: Vec<PackedSfenValue> = Vec::with_capacity(batch_size * 4);

        // Initialize RNG for random-fen-skipping
        let mut rng = SimpleRand::with_seed();

        // Flag to track if we need to refill the buffer
        let skipping_enabled = self.random_fen_skipping > 0 || self.early_fen_skipping > 0;

        // Track iterations to prevent infinite loops
        let mut iteration = 0;

        'dataloading: loop {
            // Safety check: prevent infinite loops
            if iteration >= MAX_ITERATIONS {
                eprintln!("Warning: Reached maximum iteration limit ({}) while filtering data. ", MAX_ITERATIONS);
                eprintln!("         This may indicate that too many positions are being filtered. ");
                eprintln!("         Proceeding with {} pending positions.", pending.len());
                break;
            }
            iteration += 1;

            let mut loader_files = vec![];
            for file in file_paths.iter() {
                loader_files.push(File::open(file).unwrap());
            }

            for (mut loader_file, file_path) in loader_files.into_iter().zip(file_paths.iter()) {
                if to_skip > 0 {
                    println!("Skipping to {to_skip}th entry in file [{file_path}]");
                    loader_file.seek(SeekFrom::Current((to_skip * data_size) as i64)).unwrap();
                    to_skip = 0;
                }

                loop {
                    let count = loader_file
                        .read(unsafe {
                            std::slice::from_raw_parts_mut(read_buf.as_mut_ptr() as *mut u8, cap * data_size)
                        })
                        .unwrap_or(0);

                    if count == 0 {
                        // End of this file, continue to next file
                        break;
                    }

                    assert_eq!(count % data_size, 0);
                    let len = count / data_size;

                    // Process the read buffer
                    for batch in read_buf[..len].chunks(batch_size) {
                        if skipping_enabled {
                            // Filter positions and add to pending buffer
                            for pos in batch.iter() {
                                // Early-fen-skipping: check ply
                                let early_skip_passed = if self.early_fen_skipping > 0 {
                                    pos.game_ply() >= self.early_fen_skipping as u16
                                } else {
                                    true
                                };

                                if !early_skip_passed {
                                    continue;
                                }

                                // Random-fen-skipping
                                if self.random_fen_skipping > 0 {
                                    let rand_val = (rng.rng() % (self.random_fen_skipping as u64 + 1)) as u32;
                                    if rand_val != 0 {
                                        continue;
                                    }
                                }

                                // Position passed all filters, add to pending
                                pending.push(*pos);
                            }

                            // Try to emit full batches from pending buffer
                            if Self::emit_batches(&mut pending, batch_size, &mut f) {
                                break 'dataloading;
                            }
                        } else {
                            // No skipping, process batch normally
                            let should_break = f(batch);
                            if should_break {
                                break 'dataloading;
                            }
                        }
                    }
                }
            }

            // After processing all files, check if we have remaining pending positions
            // and need to loop back to read more data
            if skipping_enabled && !pending.is_empty() {
                // We have some pending positions but not enough for a full batch
                // and we've reached the end of all files.
                // In this case, we continue the loop to read from the beginning again.
                // This ensures we maintain the target number of positions per epoch.

                // Rotate files back to start for the next iteration
                file_paths = self.file_paths.clone();

                // Emit any full batches that might have accumulated
                if Self::emit_batches(&mut pending, batch_size, &mut f) {
                    break 'dataloading;
                }

                // Continue reading from the beginning to fill the batch
                continue;
            }

            // No skipping or no pending data, we're done with this epoch
            break;
        }

        // Emit any remaining pending positions as a final partial batch
        // (this may happen if the total filtered positions don't divide evenly by batch_size)
        if skipping_enabled && !pending.is_empty() {
            f(&pending);
        }
    }
}

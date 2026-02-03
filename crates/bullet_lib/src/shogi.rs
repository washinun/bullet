//! 将棋固有のモジュール
//!
//! bullet を将棋 NNUE 学習に対応させるための型・関数定義。
//!
//! - `types`: 基本型 (Color, PieceType, Square, Piece)
//! - `packed_sfen`: PackedSfen/PackedSfenValue デコーダ
//! - `bona_piece`: BonaPiece 定義
//! - `data_loader`: 将棋用データローダー (fen-skipping 対応)

pub mod bona_piece;
pub mod data_loader;
pub mod packed_sfen;
pub mod types;

pub use bona_piece::BonaPiece;
pub use data_loader::ShogiDirectSequentialDataLoader;
pub use packed_sfen::{BitStream, PackedSfen, PackedSfenValue, ShogiBoard};
pub use types::{Color, Hand, Piece, PieceType, Square};

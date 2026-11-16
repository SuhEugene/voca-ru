# Speed comparison

Machine: Apple M1 Pro, Darwin 25.3.0 (arm64).  
Software: PyTorch 2.15.0, ONNX Runtime 1.32.0, Python 3.12.15, rustc 1.100.0 (0ef1bc386 2026-11-09).

Produced by `bash scripts/benchmark.sh`. Every engine gets the same phoneme ids; times are the median of repeated runs
after warm-up, for one utterance, model only (phonemes in, waveform out, noise generation included).

## Model only, 1 thread

| Audio | Phonemes | PyTorch (reference) | ONNX Runtime (published model) | Rust, pure-Rust matmul | Rust, Accelerate | Rust vs PyTorch | Rust vs ONNX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.9 s | 24 | 19.4 ms (100x) | 31.6 ms (61x) | 19.2 ms (101x) | 6.9 ms (283x) | **2.8x** | 4.6x |
| 4.1 s | 63 | 38.4 ms (106x) | 73.8 ms (55x) | 46.0 ms (89x) | 15.5 ms (263x) | **2.5x** | 4.7x |
| 6.3 s | 102 | 61.5 ms (102x) | 123.4 ms (51x) | 80.9 ms (78x) | 28.8 ms (218x) | **2.1x** | 4.3x |
| 16.8 s | 279 | 205.7 ms (82x) | 483.4 ms (35x) | 379.5 ms (44x) | 116.9 ms (144x) | **1.8x** | 4.1x |

(n x) is the multiple of real time. Model load: PyTorch (reference) 807 ms, ONNX Runtime (published model) 161 ms, Rust, pure-Rust matmul 12 ms, Rust, Accelerate 13 ms.

## Model only, 4 threads

| Audio | Phonemes | PyTorch (reference) | ONNX Runtime (published model) | Rust, pure-Rust matmul | Rust, Accelerate | Rust vs PyTorch | Rust vs ONNX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.9 s | 24 | 46.7 ms (42x) | 14.6 ms (133x) | 10.3 ms (188x) | 6.8 ms (287x) | **6.9x** | 2.2x |
| 4.1 s | 63 | 56.6 ms (72x) | 31.9 ms (128x) | 19.7 ms (207x) | 12.0 ms (341x) | **4.7x** | 2.7x |
| 6.3 s | 102 | 67.3 ms (93x) | 52.1 ms (120x) | 32.0 ms (196x) | 17.8 ms (353x) | **3.8x** | 2.9x |
| 16.8 s | 279 | 131.9 ms (127x) | 187.8 ms (89x) | 118.5 ms (142x) | 60.1 ms (280x) | **2.2x** | 3.1x |

(n x) is the multiple of real time. Model load: PyTorch (reference) 532 ms, ONNX Runtime (published model) 112 ms, Rust, pure-Rust matmul 5 ms, Rust, Accelerate 5 ms.

## Whole command, new process (text in, WAV file out)

`python -m tiny_voca.infer` against `tiny-voca --no-split`, one thread each on the Rust side, median of 10 runs.

| Audio | Python | Rust | Speed-up |
|---:|---:|---:|---:|
| 1.9 s | 880 ms | 24 ms | **37x** |
| 4.1 s | 891 ms | 33 ms | **27x** |
| 6.3 s | 907 ms | 45 ms | **20x** |

## Phonemizer only (dizione)

1828 test texts (heavy in numbers, dates and made-up words); 0 differ between the two implementations.

| | Python | Rust | Speed-up |
|---|---:|---:|---:|
| Load the dictionary and tables | 72.1 ms | 5.21 ms | **13.8x** |
| Per text, first pass (cold word cache) | 0.736 ms | 0.441 ms | **1.7x** |
| Per text, warm | 0.111 ms | 0.0383 ms | **2.9x** |


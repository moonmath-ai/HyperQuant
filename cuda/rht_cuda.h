#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Random Hadamard Transform (RHT) for incoherence preprocessing (QuIP#/QuaRot
// style, HIGGS-compatible block structure). The rotation is
//
//     R = (1/sqrt(H)) · D · H_block
//
// where H_block is the unnormalized ±1 Walsh–Hadamard matrix applied in
// contiguous blocks of `had_size` along the contraction dimension, and D is a
// fixed random ±1 diagonal (the "random" in RHT; HIGGS uses a plain Hadamard,
// i.e. D = I). Because R is orthonormal (R Rᵀ = I), a linear layer can keep the
// weight rotated-and-compressed and rotate the activation online:
//
//     x W = (x R)(Rᵀ W) = decode(encode(Rᵀ W)) · (x R)          (R⁻¹ = Rᵀ)
//
// Both rotations are the SAME elementwise op — y = (1/sqrt(H))·FWHT_H(in ⊙ d) per
// had_size-block, with the same sign vector d (length K) applied along the
// contraction dim K: for activations along the last dim, for weights along the
// first dim (offline). This class applies it along the last dim (the runtime
// activation path); the host signs are exposed so the offline W rotation and a
// CPU reference can use the identical d.
//
// bf16 in/out. K must be a multiple of had_size; had_size ∈ {256,512,1024,2048}.
class RhtCuda {
 public:
  // Generates K random ±1 signs from `seed` and uploads them. had_size = the
  // Hadamard block length (1024 matches HIGGS' default).
  RhtCuda(int had_size, int K, uint32_t seed);
  ~RhtCuda();

  RhtCuda(const RhtCuda&) = delete;
  RhtCuda& operator=(const RhtCuda&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

  // Apply R along the last dim of a (rows × K) row-major bf16 matrix `d_x`
  // (device ptr), writing `d_out` (device ptr, may alias d_x). Asynchronous on
  // `cuda_stream` (cudaStream_t as void*, default stream if null). No host copy.
  bool apply_lastdim(const void* d_x_bf16, void* d_out_bf16, int rows, void* cuda_stream = nullptr);

  int had_size() const { return had_size_; }
  int K() const { return K_; }
  // The K random ±1 signs (host copy), for the offline W rotation / CPU reference.
  const std::vector<int8_t>& host_signs() const { return host_signs_; }

 private:
  bool ok_ = false;
  std::string error_message_;
  int had_size_ = 0;
  int K_ = 0;
  std::vector<int8_t> host_signs_;
  int8_t* d_signs_ = nullptr;
};

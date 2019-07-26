module TruncatedGaussQuadrature

#=
To import module...
push!(LOAD_PATH, "/Path/To/Src/")
... then load the module
using TruncatedGaussQuadrature
.. Test the module with:
truncatedGaussQuad(0.0, 0.1, -Inf, 10.0, 5)
=#


import Distributions
import Combinatorics
using LinearAlgebra # Hermitian, chol, permute, eigen

# include Distributions # loads other packages that subsequent code depends ons
include("normalDistMoment.jl")
include("momentGaussQuad.jl")
include("spd.jl")

export truncatedGaussQuad, normalDistMoment, momentGaussQuad, buildHankel, spdMatrix


function truncatedGaussQuad(μ::Real, σ::Real, lb::Real, ub::Real, n::Int)

  # 1) Calculate moments
  mm = normalDistMoment(μ, σ, 2n, lb, ub)
  # 2) Build Hankel Matrix
  M = buildHankel(mm)
  # 3) Calculate nodes:
  x, w = momentGaussQuad(M)
  return x, w
end

function truncatedGaussQuad(lb::Real, ub::Real, n::Int)
  # N(0,1)
  x, w = truncatedGaussQuad(0.0,1.0,lb,ub,n)
  return x, w
end


function buildHankel(mm::Array{T,1}) where T<:Real
  n = div((length(mm)-1),2)
  M = zeros(eltype(mm),n+1,n+1)
  for i = 1:n+1
    M[:,i] = mm[i:i+n]
  end
  if  isposdef(M)
  else
      M = spdMatrix(M)
  end
  return M
end

end

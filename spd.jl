#=

Author: Benjamin Hemingway

This code adjusts the square matrix in the case it is not positive semi definite.

This code is based on the Matlab package nearestSPD version 1.1.0.0 by John D'Errico
=#
function spdMatrix(M::Array{T,2}) where T<:Real
  # symmetrize A into B
  B = (M + M')/2;
  # Compute the symmetric polar factor of B. Call it H.
  # Clearly H is itself SPD.
  out = LinearAlgebra.svd(B)
  H = out.Vt'*Diagonal(out.S)*out.Vt
  # get Ahat in the above formula
  Ahat = (B+H)/2
  # ensure symmetry
  Ahat = (Ahat + Ahat')/2;
  # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
  p = isposdef(Ahat)
  k = 0;
  while p==false
    k+=1
    # Ahat failed the chol test. It must have been just a hair off,
    # due to floating point trash, so it is simplest now just to
    # tweak by adding a tiny multiple of an identity matrix.
    mineig = minimum(eigvals(Ahat))
    Ahat += (-mineig*k.^2 + eps(mineig))*Matrix{Float64}(I,size(Ahat))
    p = isposdef(Ahat)
    if k>99
      error("Could not create Positive Definite Matrix")
    end
  end
  return Ahat
end

#=

Author: Benjamin Hemingway

This code returns the Gaussian quadrature nodes and weights associated
with a Truncated Normal distribution.

This code is based on the method outlined by John Burkardt in the following
presentation:
https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
=#


#---------------------------------------------------------------------#
# Calculate Nodes and Weights
#---------------------------------------------------------------------#


function momentGaussQuad(M::Array{T,2}) where T<:Real

  if size(M)[1]!=size(M)[2]
    error("Hankel Matrix M must be square")
  end

  # 1) Hankel matrix already constructed
  n = (size(M)[1]-1)

  # 2) Find Upper-trinagular Cholesky decomposition: M=R'*R
  # R = chol(M)
  R = (cholesky(Hermitian(M))).U

  # 3) Compute a vector α of length n and vector β of length (n-1).
  α = Array{eltype(M),1}(undef,n)
  β = Array{eltype(M),1}(undef,n-1)

  α[1] = R[1,2] / R[1,1]

  for i = 2:n
    α[i] = R[i,i+1] / R[i,i] - R[i-1,i] / R[i-1,i-1]
  end

  for i = 1:n-1
    β[i] = R[i+1,i+1] / R[i,i]
  end

  # 4) Use α and β to generate the Golub-Welsch matrix J.

  J = Array{eltype(M),2}(undef,n,n)
  for i=1:n, j=1:n
    if i==j
      J[i,j] = α[i]
    elseif i==j+1
      J[i,j] = β[j]
    elseif i==j-1
      J[i,j] = β[i]
    else
      J[i,j] = 0.0
    end
  end

  # 5) Determine the eigenvalues L and eigenvectors V of J
  #   (in matlab use: [V, L] = eig(J) )
  dcomp = eigen(J,permute=true,scale=true)

  # 6) Generate the nodes and weights using:
  #     x = diag(L)
  #     w = μ0*V(1,1:m).^2
  x = dcomp.values
  w = vec(dcomp.vectors[1,1:n].^2)

  #= Weights are unadjusted...
  # 7) adjust weights
  dist = Distributions.Normal(μ,σ)
  Gub = Distributions.cdf(dist,ub)
  Glb = Distributions.cdf(dist,lb)

  return x, (Gub-Glb)*w
  =#
  return x, w
end

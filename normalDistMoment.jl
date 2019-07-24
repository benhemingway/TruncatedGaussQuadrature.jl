#=

Author: Benjamin Hemingway

This code returns the Gaussian quadrature nodes and weights associated
with a Truncated Normal distribution.

This code is based on the method outlined by John Burkardt in the following
presentation:
https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
=#
#---------------------------------------------------------------------#
# Generate Moments
#---------------------------------------------------------------------#
function normalDistMoment(μ::Real,σ::Real, n::Int)
  _normalDistMoment(μ, σ, n, -Inf, Inf)
end
function normalDistMoment(μ::Real,σ::Real, n::Int, l::Float64, u::Float64)
  _normalDistMoment(μ, σ, n, l, u)
end
function normalDistMoment(n::Int)
  _normalDistMoment(n, -Inf, Inf)
end
function normalDistMoment(n::Int, l::Float64, u::Float64)
  _normalDistMoment(n, l, u)
end

function _normalDistMoment(μ::Real,σ::Real, n::Int, l::Real, u::Real)

  f0 = Distributions.Normal(0,1)
  mom = Array{Float64,1}(undef,n+1)
  mom[1] = 1.0
  for i = 1:n
    mom[i+1]=0.0
  end

  if n<1
    Void
  end

  if l>-Inf && u<Inf # If the distribution is doubly truncated
    α = (l - μ)/σ
    β = (u - μ)/σ

    # Calculate the recursive part of moments
    L = Array{Float64,1}(undef,n+1)
    L[1] = 1.0
    L[2] = ( Distributions.pdf(f0,α)-Distributions.pdf(f0,β)
    ) /  ( Distributions.cdf(f0,β)-Distributions.cdf(f0,α) )

    for i = (3:n+1)
      L[i] = (( α^(i-2)*Distributions.pdf(f0,α)-β^(i-2)*
      Distributions.pdf(f0,β) ) / ( Distributions.cdf(f0,β)
      -Distributions.cdf(f0,α) ) + (i-2)*L[i-2])
    end

    for k = 1:n, i =0:k
      mom[k+1] = (binomial(k,i)*L[i+1])*(σ^i)*(μ^(k-i)) + mom[k+1]
    end

  elseif l>-Inf && u==Inf # If the distribution is upper truncated
    α = (l - μ)/σ
    L = Array{Float64,1}(undef,n+1)
    L[1] = 1.0
    L[2] = ( Distributions.pdf(f0,α) )/( Distributions.ccdf(f0,α) )
    for i = (3:n+1)
      L[i] = (((α^(i-2))*Distributions.pdf(f0,α))/(Distributions.ccdf(f0,α))+(i-2)*L[i-2])
    end

    for k = 1:n, i =0:k
      mom[k+1] = ((binomial(k,i)*L[i+1])*(σ^i)*(μ^(k-i)) + mom[k+1])
    end

  elseif l==-Inf && u<Inf # If the distribution is lower truncated
    L = Array{Float64,1}(undef,n+1)
    β = (u - μ)/σ

    L[1] = 1.0
    L[2] = ( -Distributions.pdf(f0,β) )/( Distributions.cdf(f0,β) )

    for i = (3:n+1)
      L[i] = (-(β^(i-2))*Distributions.pdf(f0,β) / Distributions.cdf(f0,β)  + (i-2)*L[i-2])
    end

    for k = 1:n, i =0:k
      mom[k+1] = ((binomial(k,i)*L[i+1])*(σ^i)*(μ^(k-i)) + mom[k+1])
    end

  else # If the distribution is non-truncated
    mom[2] = μ
    for i = 3:n+1
      ind = floor(Int,(i-1)/2)
      for j = 0:ind
        if 2j-1<0
          # Convention is (-1)!!=1 but not followed by Combinatorics.doublefactorial
          mom[i] += binomial(i-1,2j)*σ^(2j)*μ^(i-2j-1)
        else
          mom[i] += binomial(i-1,2j)*σ^(2j)*μ^(i-2j-1)*Combinatorics.doublefactorial(2j-1)
        end
      end
    end
  end
  # Return the vector of moments
  return mom
end

function _normalDistMoment(n::Int, α::Real, β::Real)
  # calculation for standard normal
  f0 = Distributions.Normal(0,1)
  mom = zeros(n+1)
  mom[1] = 1.0

  if n<1
    Void
  end

  if α>-Inf && β<Inf # If the distribution is doubly truncated

    # Calculate the recursive part of moments
    L = Array{Float64,1}(undef,n+1)
    L[1] = 1.0
    L[2] = ( Distributions.pdf(f0,α)-Distributions.pdf(f0,β)
    ) /  ( Distributions.cdf(f0,β)-Distributions.cdf(f0,α) )

    for i = (3:n+1)
      L[i] = (( α^(i-2)*Distributions.pdf(f0,α)-β^(i-2)*
      Distributions.pdf(f0,β) ) / ( Distributions.cdf(f0,β)
      -Distributions.cdf(f0,α) ) + (i-2)*L[i-2])
    end

    for k = 1:n
      mom[k+1] += L[k+1]
    end

  elseif α>-Inf && β==Inf # If the distribution is upper truncated
    L = Array{Float64,1}(undef,n+1)
    L[1] = 1.0
    L[2] = ( Distributions.pdf(f0,α) )/( Distributions.ccdf(f0,α) )
    for i = (3:n+1)
      L[i] = (((α^(i-2))*Distributions.pdf(f0,α))/(Distributions.ccdf(f0,α))+(i-2)*L[i-2])
    end

    for k = 1:n
      mom[k+1] += L[k+1]
    end

  elseif α==-Inf && β<Inf # If the distribution is lower truncated
    L = Array{Float64,1}(undef,n+1)

    L[1] = 1.0
    L[2] = ( -Distributions.pdf(f0,β) )/( Distributions.cdf(f0,β) )

    for i = (3:n+1)
      L[i] = (-(β^(i-2))*Distributions.pdf(f0,β) / Distributions.cdf(f0,β)  + (i-2)*L[i-2])
    end

    for k = 1:n
      mom[k+1] += L[k+1]
    end

  else # If the distribution is non-truncated
    mom[2] = 0.0
    for i=3:n+1
      if rem(i-1,2)==0
        mom[i] += Combinatorics.doublefactorial(i-2)
      end
    end
  end

  # Return the vector of moments
  return mom

end

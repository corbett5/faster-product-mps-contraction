using MKL
using ITensorMPS
using ITensors

struct DeconstructedMPS{C, N}
  left_tensor::Array{C, 2}
  middle_tensors::Vector{Array{C, 3}}
  right_tensor::Array{C, 2}
  work::Matrix{C}
end

struct CompileTimeInt{N}
end

#=
Take a MPS phi that has elements of type C and for which each site has local dimension N (N = 2 is a qubit, N = 3 is spin 1...)
and do some preperation for taking lots of inner products.
=#
function prepare_for_lots_of_inner_products(::Type{C}, ::CompileTimeInt{N}, phi::MPS)::DeconstructedMPS{C, N} where {C, N}
  phi = dag(phi)
  sim!(linkinds, phi)

  links = linkinds(phi)
  sites = siteinds(phi)

  @assert all(dim(site) == N for site in sites)

  left_tensor = array(phi[1], sites[1], links[1])

  middle_tensors = Array{C, 3}[]
  for n in 2:(length(phi) - 1)
    push!(middle_tensors, array(phi[n], sites[n], links[n - 1], links[n]))
  end

  right_tensor = array(phi[end], sites[end], links[end])

  return DeconstructedMPS{C, N}(left_tensor, middle_tensors, right_tensor, zeros(C, maxlinkdim(phi), 2))
end

#=
Basically a copy of Julia's built in dot that allows for compile time knowledge of the length of x as well as
removes a bunch of checks.
=#
function my_dot(::CompileTimeInt{N}, x::AbstractVector{C}, A::AbstractMatrix{C}, y::AbstractVector{C}) where {N, C}
  s = zero(C)
  @inbounds for j in eachindex(y)
    temp = zero(C)
    @simd for i in 1:N
      temp += adjoint(A[i,j]) * x[i]
    end
    s += y[j] * temp
  end

  return s
end

#=
Compute the inner product between phi and a product MPS p, p[:, i] is the state of the ith sub-system.
=#
function inner_product(phi::DeconstructedMPS{C, N}, p::Matrix{C})::Number where {C, N}
  work = phi.work

  @inbounds for rl in 1:size(phi.left_tensor, 2)
    work[rl, 1] = dot(view(p, :, 1), view(phi.left_tensor, :, rl))
  end

  cur_work, next_work = 1, 2
  @inbounds for n in 1:length(phi.middle_tensors)
    tensor = phi.middle_tensors[n]
    for rl in 1:size(tensor, 3)
      work[rl, next_work] = my_dot(CompileTimeInt{N}(), view(p, :, n + 1), view(tensor, :, :, rl), view(work, 1:size(tensor, 2), cur_work))
    end

    cur_work, next_work = next_work, cur_work
  end

  return @inbounds dot(view(p, :, size(p, 2)), phi.right_tensor, view(work, 1:size(phi.right_tensor, 2), cur_work))
end


###############################################################################
# Everything below here is for testing/benchmarking
###############################################################################

function prepare(N::Int, local_dim::Int, link_dim::Int)::Tuple{MPS, MPS, DeconstructedMPS{Float64, local_dim}, Matrix{Float64}}
  sites = siteinds("Qudit", N; dim=local_dim)

  phi = random_mps(sites, linkdims=link_dim)
  p = random_mps(sites)

  phi_decomposed = prepare_for_lots_of_inner_products(eltype(phi[1]), CompileTimeInt{local_dim}(), phi)

  p_vecs = [reshape(array(t), local_dim) for t in p]
  p_matrix = zeros(local_dim, N)

  for n in 1:N
    p_matrix[:, n] .= p_vecs[n]
  end

  return phi, p, phi_decomposed, p_matrix
end

function test_inner_product(N::Int, local_dim::Int, link_dim::Int)::Nothing
  phi, p, phi_decomposed, p_matrix = prepare(N, local_dim, link_dim)

  value_from_itensor = inner(phi, p)
  my_value = inner_product(phi_decomposed, p_matrix)
  relative_diff = abs(value_from_itensor - my_value) / abs(value_from_itensor)
  @assert relative_diff < 1e-10 relative_diff

  return nothing
end

function benchmark_inner_product(N::Int, local_dim::Int, link_dim::Int, n_iter::Int)::Nothing
  phi, p, phi_decomposed, p_matrix = prepare(N, local_dim, link_dim)

  println("Contracting an MPS on $N sites each with local dim $local_dim and a max link dim of $link_dim, $n_iter times")

  @time "ITensor contractions" for _ in 1:n_iter
    inner(phi, p)
  end

  @time "my contractions" for _ in 1:n_iter
    inner_product(phi_decomposed, p_matrix)
  end

  println()

  return nothing
end


for N in 2:10
  for local_dim in 2:8
    for link_dim in 1:20
      test_inner_product(N, local_dim, link_dim)
    end
  end
end

benchmark_inner_product(20, 3, 10^1, 10^5)
benchmark_inner_product(20, 3, 10^2, 10^4)
benchmark_inner_product(20, 3, 10^3, 10^3)

###############################################################################
# Benchmarks ran on my desktop
###############################################################################
#=
Contracting an MPS on 20 sites each with local dim 3 and a max link dim of 10, 100000 times
ITensor contractions: 15.318093 seconds (348.90 M allocations: 120.489 GiB, 9.43% gc time)
my contractions: 0.151394 seconds

Contracting an MPS on 20 sites each with local dim 3 and a max link dim of 100, 10000 times
ITensor contractions: 5.760894 seconds (34.89 M allocations: 12.475 GiB, 2.89% gc time)
my contractions: 0.795498 seconds

Contracting an MPS on 20 sites each with local dim 3 and a max link dim of 1000, 1000 times
ITensor contractions: 6.192408 seconds (3.50 M allocations: 1.598 GiB, 1.64% gc time)
my contractions: 17.601998 seconds
=#

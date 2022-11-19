module SparseTensors

export SparseTensor, ptranspose , ptrace, eigenvalues, permute

using LinearAlgebra

mutable struct SparseTensor{Tv,Ti}
    is::Vector{Vector{Vector{Ti}}} #Vector of coords, which must all be the same length
    vs::Vector{Tv}
    shape:: Vector{Int}
    
    function SparseTensor{Tv,Ti}(is::Vector{Vector{Vector{Ti}}}, vs::Vector{Tv} ,shape:: Vector{Int}) where {Ti,Tv} #is: is[i] is the index for the i-th element. is[i][1/2] left/right set of indices for the i-th element. is[i][1/2][j] gives the actual left/right indices for the j-th subspace for the i-th element.
        length(is) == length(vs) ||
            throw(ArgumentError("Index vector and data should be of equal length."))
        all(length.(is) .== 2) ||
            throw(ArgumentError("All index vectors should be given in pairs of [[left],[right]]."))
        all(all(length.(is[i]) .== length(shape)) for i ∈ eachindex(is)) ||
            throw(ArgumentError("All index vectors should be of size (shape)."))
        all(1 <= is[i][k][j] <= shape[j] for i ∈ eachindex(is), j ∈ eachindex(shape), k ∈ 1:2) ||
            throw(ArgumentError("Index out of bounds relative to shape"))
        new{Tv,Ti}(is, vs, shape)
    end
end

SparseTensor(is::Vector{Vector{Vector{Ti}}}, vs::Vector{Tv} ,shape:: Vector{Int}) where {Ti,Tv} =
SparseTensor{Tv,Ti}(is, vs ,shape)

Base.size(A::SparseTensor) = return A.shape

function Base.getindex(A::SparseTensor{Tv,Ti}, inds::Vector{Vector{Ti}}) where {Tv,Ti}
    res = findfirst(isequal(inds),A.is)
    if isnothing(res)
        return res
    end
    return A.vs[res]
end

function Base.setindex!(A::SparseTensor{Tv,Ti}, v, inds::Vector{Vector{Ti}}) where {Tv,Ti}
    all(length.(inds)==length(A.shape)) && all(1 <= inds[j][i] <= shape[i] for i ∈ eachindex(shape), j ∈ 1:2) || throw(BoundsError(A, inds))
    idx = findfirst(isequal(inds),A.is)
    if isnothing(idx) # index isn't in tensor, make new element
        push!(A.is, inds)
        push!(A.vs, convert(Tv, v))
    else
        A.vs[idx] = convert(Tv,v)
    end
    return A
end

function permute(A::SparseTensor{Tv,Ti},perm::AbstractVector{<:Integer}) where {Ti,Tv} # Can only permute subsystems, can't partial transpose.
    (isperm(perm) && maximum(perm)==length(A.shape)) || throw(ArgumentError("permutation vector is not a valid permutation"))
    newDims = similar(A.is)
    for i ∈ eachindex(A.is)
        for j ∈ 1:2
        permutedims!(newDims[i],A.is[j][i],perm)
        end
    end
end

function ptranspose(A::SparseTensor{Tv,Ti},ind::Integer) where {Ti,Tv}
    ind>=1 && ind<=length(A.shape) ||
        throw(ArgumentError("Transposition index not in range"))
    
    for i ∈ eachindex(A.is)
        temp = A.is[i][2][ind]
        A.is[i][2][ind] = A.is[i][1][ind]
        A.is[i][1][ind] = temp
    end
end

function ptrace(A::SparseTensor{Tv,Ti},ind::Integer) where {Ti,Tv}
    ind>=1 && ind<=length(A.shape) ||
        throw(ArgumentError("Transposition index not in range"))
    
    remainingTerms = findall(x->x[1][ind]==x[2][ind],A.is) # find terms that are on the diagonal with respect to subsystem number "ind". Everything else goes to zero.
    if isempty(remainingTerms)
        @warn "Partial trace returned a zero matrix. Returning scalar 0"
        return 0
    end
    n = length(remainingTerms)
    newIs = deepcopy(A.is[remainingTerms])
    newVs = deepcopy(A.vs[remainingTerms])
    newDims = reduce(vcat,collect.([1:(ind-1),(ind+1):length(shape)]))
    newShape = A.shape(newDims)
    
    for i ∈ 1:n
        for j ∈ 1:2
        newIs[i][j] = newIs[i][j][newDims]
        end
    end
    uniqueSet = Dict{eltype(newIs),Vector{Ti}}
    for i ∈ 1:n
        if newIs[i] ∈ uniqueSet.keys
            push!(uniqueSet[newIs[i]],i)
        else
            uniqueSet[newIs] = [i]
        end
    end

    inds = [vals for vals ∈ values(uniqueSet)] # Done so I don't have to worry about whether dictionaries return values in the same order each time.
    newIs = newIs[[inds[:][1]]]
    newVs = [sum(newVs[i]) for i ∈ inds]

    if length(A.shape)==1
        @warn "Partial trace over a single dimension is a full trace, return result is a scalar"
        return newVs[1]
    end
    return SparseTensor(newIs,newVs,newShape)
end

function strides(A::SparseTensor{Tv,Ti}) where {Ti,Tv}
    return [1,cumprod(A.shape)[1:end-1]...]
end



function toMatrix(A::SparseTensor{Tv,Ti}) where {Ti,Tv}
    s = strides(A)
    Is = [sum((A.is[i][1] .- 1).*s) + 1 for i=1:length(A.is)]
    Js = [sum((A.is[i][2] .- 1).*s) + 1 for i=1:length(A.is)]
    n = prod(size(A))
    return Is,Js
end

function eigenvalues(A::SparseTensor{Tv,Ti}) where {Ti,Tv} #throws away all rows and columns with nothing in them, then hands the submatrix off to a solver
    Is,Js = toMatrix(A)
    uniqueIs = unique(Is)
    uniqueJs = unique(Js)
    for (i,fullind) in enumerate(uniqueIs)
        Is[Is.==fullind] .= i
    end
    for (j,fullind) in enumerate(uniqueJs)
        Js[Js.==fullind] .= j
    end
    n = maximum([length(uniqueIs),length(uniqueJs)])
    Mat = zeros(Tv,n,n)
    Mat[Is,Js] .= A.vs
    return eigvals(Mat)
end
end

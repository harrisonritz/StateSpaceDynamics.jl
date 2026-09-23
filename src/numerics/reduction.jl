#=============================================================================
Trial chunking for parallel reductions.

A parallel sum over trials is split into chunks, each accumulated in trial order
and then added into the total in chunk order. Its floating-point result is set by
where the chunk boundaries fall, so they are fixed by the trial count alone —
never by the thread count, the workspace pool or `ntasks` — and a fit gives the
same bits whatever machine it runs on. The number of buffers only decides how
many chunks are in flight at once: `_foreach_chunk_wave` runs the chunks in waves
of that many and hands each wave back to be reduced, in chunk order, before the
buffers are reused.
=============================================================================#

# Upper bound on the chunk count. Beyond it a reduction runs no more chunks in
# parallel than this, whatever the thread count.
const _REDUCTION_CHUNKS = 32

"""
    _reduction_chunks(n) -> Vector{UnitRange{Int}}

`1:n` as at most `_REDUCTION_CHUNKS` contiguous chunks of near-equal size, a
function of `n` alone.
"""
function _reduction_chunks(n::Int)
    n <= 0 && return UnitRange{Int}[]
    size = cld(n, min(n, _REDUCTION_CHUNKS))
    return [((i - 1) * size + 1):min(i * size, n) for i in 1:cld(n, size)]
end

"""
    _foreach_chunk_wave(accumulate!, reduce!, chunks, nbuf)

Run `accumulate!(slot, chunk)` over `chunks` in parallel waves of at most `nbuf`,
buffer `slot` holding chunk `chunk`'s partial, and after each wave call
`reduce!(slot)` for its chunks in chunk order. Every chunk is accumulated on its
own and the partials are reduced strictly in chunk order, so the total does not
depend on `nbuf`.
"""
function _foreach_chunk_wave(accumulate!, reduce!, chunks::AbstractVector, nbuf::Int)
    nbuf >= 1 || throw(ArgumentError("need at least one buffer, got $nbuf"))
    for wave in Iterators.partition(eachindex(chunks), nbuf)
        tforeach(eachindex(wave)) do slot
            accumulate!(slot, chunks[wave[slot]])
            return nothing
        end
        for slot in eachindex(wave)
            reduce!(slot)
        end
    end
    return nothing
end

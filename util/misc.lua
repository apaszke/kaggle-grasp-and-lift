
-- misc utilities

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function sliceTable(table, to)
    local slice = {}
    for i = 1, to do
        slice[i] = table[i]
    end
    return slice
end

function calculate_avg_loss(losses)
    local smoothing = 40
    local sum = 0
    for i = #losses, math.max(1, #losses - smoothing + 1), -1 do
        sum = sum + losses[i]
    end
    return sum / math.min(smoothing, #losses)
end

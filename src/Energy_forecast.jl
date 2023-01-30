module Energy_forecast

using Flux, XLSX, DataFrames, Glob, Plots

export standardize, is_best

include("data_processing.jl")
include("metrics.jl")

end

# Run https://github.com/legend-exp/legend-julia-tutorial first, then

using SparseArrays, LinearAlgebra, HDF5

sel = findall(x -> x >= 50u"keV", sum.(filtered_events.edep))
sel_events = filtered_events[sel]

ssd_signals_allch = simulate_waveforms(      
    sel_events,
    sim,
    max_nsteps = 2000, 
    Δt = 1u"ns", 
    verbose = false
)

ssd_signals = ssd_signals_allch[findall(isequal(1), ssd_signals_allch.chnid)]

WF_fullres = zeros(1024, length(ssd_signals))
for (sigout, wf) in zip(eachcol(WF_fullres), ssd_signals.waveform)
    signal = ustrip.(add_baseline_and_extend_tail(wf, 2000, 6000).signal)
    wfmax = maximum(signal)
    itrig = something(findfirst(x -> x > wfmax/2, signal), 2000)
    sigview = view(signal, (itrig-512):(itrig+511))

    copyto!(sigout, sigview)
end

flt_idxs = findall(x -> x>1, WF_fullres[end,:])

flt_events = sel_events[flt_idxs]
edep = ustrip.(sum.(flt_events.edep))
sse = map(x -> x < 1, (x -> isnan(x) ? zero(x) : x).(sqrt.(ustrip.(sum.(var.(flt_events.pos))))))

# Subsampling:
WF = sparse([ div(j-1,30) == i-1 for i in 1:32, j in 1:1024]) * WF_fullres[:,flt_idxs]

#=
using Plots
plot(WF[:,1:100])
=#

file = h5open("wfdata.h5", "w")
file["WF"] = WF
file["edep"] = edep
file["sse"] = sse
close(file)

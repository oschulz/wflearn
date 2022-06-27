using HDF5
using Random, LogExpFunctions, Distributions
using CompositionsBase, Zygote, Optimisers
using Base.Iterators: partition
using Functors: @functor, functor, fmap
using Plots, BenchmarkTools, ProgressMeter


# Read data:

file = h5open("wfdata.h5")
WF = read(file["WF"])
edep = read(file["edep"])
sse = read(file["sse"])
close(file)

plot(WF[:,findall(sse)][:,1:100])
plot(WF[:,findall(.! sse)][:,1:100])
stephist(edep, nbins = 0:5:3000, yscale = :log10)


# Normalize waveforms:
X = WF ./ maximum(WF, dims = 1)
L = sse


plot(X[:,1:100])


# Define dense ML layer:

struct DenseLayer{A,B,F}
    a::A
    b::B
    f::F
end

function DenseLayer(n_in::Integer, n_out::Integer, f::Function)
    weight_scale = sqrt(6 / (n_out + n_in))
    weights = rand(Uniform(-weight_scale, weight_scale), n_out, n_in) # glorot init
    bias = rand(n_out)
    f_sigma = f
    DenseLayer(weights, bias, f_sigma)
end

@functor DenseLayer

(l::DenseLayer)(x) = l.f.(muladd(l.a, x, l.b))


# Define model:

# actfun(x) = max(0, x)
# actfun(x) = log(1 + exp(x))
actfun(x) = x / (1 + exp(-x))


model = opcompose(
    DenseLayer(32, 8, actfun),
    DenseLayer(8, 4, actfun),
    DenseLayer(4, 1, logistic),
    vec
)

Y = model(X)

#stephist(Y, nbins = 100)


# Define loss:

xentropy(p::Real, q::Real) = -(p * log(q) + (1-p) * log(1-q))
xentropy(p::Bool, q::Real) = - ifelse(p, log(q), log(1-q))
xentropy(p::AbstractVector{<:Real}, q::AbstractVector{<:Real}) = mean(xentropy.(p, q))

loss = Base.Fix1(xentropy, L)

# cross-entropy is equivalent to negative mean likelihood:
loss(Y) ≈ mean(.- logpdf.(Bernoulli.(vec(Y)), L))


# Gradient calculation:

grad_model(model, loss, X) = Zygote.gradient((m,x) -> loss(m(x)), model, X)[1]

function loss_grad_model(model, loss, X)
    l, pullback = Zygote.pullback((m,x) -> loss(m(x)), model, X)
    d_model = pullback(one(l))[1]
    return l, d_model
end

grad_model(model, loss, X)
loss_grad_model(model, loss, X)


# Define gradient descent optimizer:

struct GradientDecent{T}
    rate::T
end

(opt::GradientDecent)(x, ::Nothing) = x
(opt::GradientDecent)(x::Real, dx::Real) = x - opt.rate * dx
(opt::GradientDecent)(x::AbstractArray, dx::AbstractArray) = x .- opt.rate .* dx
function (opt::GradientDecent)(x, dx)
    content_x, re = functor(x)
    content_dx, _ = functor(dx)
    re(map(opt, content_x, content_dx))
end

optimizer = GradientDecent(1)

optimizer(model, grad_model(model, loss, X)) isa typeof(model)


# Split dataset:

L_train = L[begin:10000]
L_test = L[10001:end]
X_train = X[:,begin:10000]
X_test = X[:,10001:end]


# Train model, unbatched:

orig_model = deepcopy(model)

model = deepcopy(orig_model)
loss_train = Base.Fix1(xentropy, L_train)
loss_history = zeros(0)
optimizer = GradientDecent(0.025)
@showprogress for i in 1:1000
    l, d_model = loss_grad_model(model, loss_train, X_train)
    push!(loss_history, l)
    model = optimizer(model, d_model)
end
plot(loss_history)


# Train model, using batches and learning rate schedule:

model = deepcopy(orig_model)
loss_history = zeros(0)
for optimizer in GradientDecent.([0.1, 0.025, 0.01, 0.0025, 0.001, 0.00025])
        @showprogress for i in 1:250
        perm = shuffle(eachindex(L_train))
        shuffled_X = X_train[:, perm]
        shuffled_L = L_train[perm]
        batchsize = 200
        bath_loss_history = zeros(0)
        for idxs in partition(eachindex(shuffled_L), batchsize)
            X_batch = view(shuffled_X, :, idxs)
            L_batch = view(shuffled_L, idxs)
            loss_batch = Base.Fix1(xentropy, L_batch)
            l, d_model = loss_grad_model(model, loss_batch, X_batch)
            push!(bath_loss_history, l)
            model = optimizer(model, d_model)
        end
        push!(loss_history, mean(bath_loss_history))
    end
end
plot(loss_history)


# Evaluate trained model:

Y = model(X)
threshold = 0:0.01:1
TPR = [count((Y .>= t) .&& L) / count(L) for t in threshold]
FPR = [count((Y .>= t) .&& .! L) / count(L) for t in threshold]
Y_thresh = Y .>= 0.5

plot(
    begin
        stephist(L, nbins = 100, normalize = true, label = "Truth")
        stephist!(model(X_train), nbins = 100, normalize = true, label = "Training pred.")
        stephist!(model(X_test), nbins = 100, normalize = true, label = "Test pred.")
    end,
    begin
        plot(threshold, TPR, label = "TPR", color = :green, xlabel = "treshold")
        plot!(threshold, FPR, label = "FPR", color = :red)
    end,
    plot(FPR, TPR, label = "ROC", xlabel = "FPR", ylabel = "TPR"),
    begin
        stephist(edep, nbins = 1500:5:1700, label = "all", xlabel = "E [keV]")
        stephist!(edep[findall(L)], nbins = 1500:5:1700, label = "label SSE")
        stephist!(edep[findall(Y_thresh)], nbins = 1500:5:1700, label = "model SSE")
    end
)


#=
# Running on a GPU:

using CUDA
cu_model = fmap(cu, model)
cu_loss = fmap(cu, loss)
cu_X = cu(X)

grad_model(cu_model, cu_loss, cu_X)
loss_grad_model(cu_model, cu_loss, cu_X)
optimizer(cu_model, grad_model(cu_model, cu_loss, cu_X))

@benchmark loss_grad_model($model, $loss, $X)
@benchmark loss_grad_model($cu_model, $cu_loss, $cu_X)
=#

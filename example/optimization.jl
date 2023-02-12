using Optim
using Flux.Optimise
using Flux.Tracker
using  Zygote, Optim, FluxOptTools, Statistics

f(x) = (x[1] - 2)^2 + (x[2] + 3)^2

# Initial guess
x0 = [0.0, 0.0]
y0 = [0.0]

params
# Call the BFGS function
result = optimize(f, x0, BFGS())

println("The minimum occurs at: ", result.minimizer)
println("The minimum value is: ", result.minimum)

#x0 dovranno essere i miei pesi
#f Loss
#result = optimize(f, x0, BFGS())
pro(x) = Flux.Losses.mse(model(x), Y_test)
pro(X_test)
params
using  Zygote, Optim, FluxOptTools, Statistics
m      = Chain(Dense(1,3,tanh) , Dense(3,1))
x      = LinRange(-pi,pi,100)'
y      = sin.(x)
loss() = mean(abs2, m(x) .- y)
loss() = Flux.Losses.mse(model(X), Y);
Zygote.refresh()
pars   = Flux.params(m)
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true,show_every = 1))

for iter  in res.iterations
   @show res.iterations
   @show lossfun
end
p0
best_params = res.minimizer
Optim.minimizer(res)
reshape(res)
Flux.loadparams!(model, param_opt);
model
@show res.minimizer[:,1] 
summary(res)






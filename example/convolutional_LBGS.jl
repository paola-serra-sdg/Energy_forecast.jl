#------CNN with LBFGS optimizer-------------
using FluxOptTools
using Optim

Random.seed!(25)
N = 24 * 4
model_CNN_lbfgs = Chain(Conv((N,),1 => 4, pad = (N-1,0),sigmoid),
                Conv((N,),4 => 4, pad = (N-1,0),sigmoid),
                Conv((N,),4 => 4, pad = (N-1,0),sigmoid),
                 Conv((N,),4 => 1, pad = (N-1,0)))   |> f64;  
            #Dense(32925, 672)) 

            


optimiser = ADAM(0.05);

params_CNN_lbfgs = Flux.params(model_CNN_lbfgs);

best_params_CNN_lbfgs = Float32[]

loss() = Flux.Losses.mse(model_CNN_lbfgs(X_train), Y_train);

lossfun, gradfun, fg!, p0 = optfuns(loss, params_CNN_lbfgs)
res_CNN= Optim.optimize(Optim.only_fg!(fg!), p0,Optim.Options(iterations=10, store_trace=true))
best_params_CNN_lbfgs = res_CNN.minimizer


copy!(params_CNN_lbfgs, best_params_CNN_lbfgs)

Flux.loadparams!(model_CNN_lbfgs, params_CNN_lbfgs);




ŷ_CNN_lbfgs = (model_CNN_lbfgs(X_test)[:].*s1).+m1
ŷ_CNN_lbfgs_st = model_CNN_lbfgs(X_test)[:]





#------CNN with LBFGS optimizer-------------
using FluxOptTools
using Optim

N = 24 * 4
model_CNN2_lbfgs = Chain(Conv((N,),1 => 8, pad = (N-1,0),sigmoid),
                    Conv((N,),8 => 15, pad = (N-1,0),sigmoid),
                    Conv((N,),15 => 3, pad = (N-1,0),sigmoid),
                    Conv((N,),3 => 1, pad = (N-1,0)))    |> f64;
            #Dense(32925, 672)) 

            


optimiser = ADAM(0.01);

params_CNN2_lbfgs = Flux.params(model_CNN2_lbfgs);

best_params_CNN2_lbfgs = Float32[]

loss() = Flux.Losses.mse(model_CNN2_lbfgs(X_train), Y_train);

lossfun, gradfun, fg!, p0 = optfuns(loss, params_CNN2_lbfgs)
res_CNN2 = Optim.optimize(Optim.only_fg!(fg!), p0,Optim.Options(iterations=10, store_trace=true))
best_params_CNN2_lbfgs = res_CNN2.minimizer


copy!(params_CNN2_lbfgs, best_params_CNN2_lbfgs)

Flux.loadparams!(model_CNN2_lbfgs, params_CNN2_lbfgs);




ŷ_CNN2_lbfgs = (model_CNN2_lbfgs(X_test)[:].*s1).+m1
ŷ_CNN2_lbfgs_st = model_CNN2_lbfgs(X_test)[:]
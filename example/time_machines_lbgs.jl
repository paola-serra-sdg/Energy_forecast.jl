using ParametricMachinesDemos
using FluxOptTools
using Optim



Random.seed!(25)
dimensions = [1, 4 , 4 , 4];#, 16, 32];


machine = RecurMachine(dimensions, sigmoid; pad = 24*4, timeblock = 4*24*2)


model_PM_lbfgs = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

params_PM_lbfgs = Flux.params(model_PM_lbfgs);




#best_paras = best_parameters( "LBFGS",model,2, X, Y, X_test,Y_test)




# Loss function
loss() = Flux.Losses.mse(model_PM_lbfgs(X_train), Y_train);

lossfun, gradfun, fg!, p0 = optfuns(loss, params_PM_lbfgs)
res_PM_lbfgs = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true))
best_params_PM_lbfgs  = res_PM_lbfgs.minimizer


copy!(params_PM_lbfgs, best_params_PM_lbfgs)


Flux.loadparams!(model_PM_lbfgs, params_PM_lbfgs);
#copy flattened optimized params 


ŷ_PM_lbfgs_st = model_PM_lbfgs(X_test)[:]
ŷ_PM_lbfgs = (model_PM_lbfgs(X_test)[:].*s1).+m1


plot( y , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_PM_lbfgs ,alpha = 0.4, lab= "ŷ PM", lw=2) 
plot!( ŷ_CNN_lbfgs, alpha = 0.4, lab= "ŷ CNN", lw=2)
plot!( ŷ_CNN2_lbfgs, alpha = 0.4, lab= "ŷ CNN 2", lw=2)
#title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast_lfbgs.pdf");
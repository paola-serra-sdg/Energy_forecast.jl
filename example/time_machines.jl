using ParametricMachinesDemos
using Optim
using FluxOptTools 


#splitting in train and test
X = st[1:end-672,:,:]
Y = st[673:end,:,:]

X_test = X[end-671:end,:,:]
Y_test = Y[end-671:end,:,:]
#IL TEST
#ultimi 672 della prediction e di Y NORMALE calcola loss on test 

data = DataLoader((X, Y) ; batchsize = 1)


dimensions = [1, 4 , 4 , 4];#, 16, 32];


machine = RecurMachine(dimensions, sigmoid; pad = 24*4, timeblock = 4*24*2)

model_PM = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

params_PM = Flux.params(model_PM);
opt = ADAM(0.1);


best_paras = best_parameters( "LBFGS",model,2, X, Y, X_test,Y_test)




loss(x, y) = Flux.Losses.mse(model_PM(x), y)
# Loss function
loss() = Flux.Losses.mse(model_PM(X), Y);

lossfun, gradfun, fg!, p0 = optfuns(loss, params)
res_PM = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true))
best_params_PM = res_PM.minimizer


copy!(params_PM, best_params_PM)

Flux.loadparams!(model, params);



epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
best_params_PM = Float32[]



for epoch in 1:200
    # Train
    Flux.train!(loss, params_PM, data,opt)

    # Saving loss
    push!(epochs, epoch)
    push!(loss_on_train, loss(X, Y))
    push!(loss_on_test,  loss(X_test, Y_test))
    @show epoch
    @show loss(X, Y)
    @show loss(X_test, Y_test)

    # Saving the best parameters
    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
        #if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
            best_params_PM = params_PM
        end
    end
end


# Extract and add new trained parameters
if isempty(best_params_PM)
    best_params_PM = params_PM
end


Flux.loadparams!(model_PM, best_params_PM);
#copy flattened optimized params 


ŷ_PM_st = model_PM(X_test)[:]
ŷ_PM = (model_PM(X_test)[:].*s1).+m1


# Visualization
plot(epochs, loss_on_train, lab="Training", c=:blue, lw=2, ylims = (0,5));
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2, ylims = (0,5));
title!("Recurrent parametric machine architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("recurrentPM_loss.png");




#plotting y predicted vs y true standaridized
plot( y_st , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_PM_st ,alpha = 0.4, lab= "ŷ PM", lw=2) 
plot!( ŷ_RNN_st , alpha = 0.4, lab= "ŷ CNN", lw=2)
title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast_st.png");



#plotting y predicted vs y true
plot( y , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_PM ,alpha = 0.4, lab= "ŷ PM", lw=2) 
plot!( ŷ_CNN, alpha = 0.4, lab= "ŷ CNN", lw=2)
title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast.png");


maximum(ŷ_RNN )
minimum(ŷ_RNN)

maximum(ŷ_PM )
minimum(ŷ_PM)
using ParametricMachinesDemos


#splitting in train and test
#X = st[1:end-672,:,:]
#Y = st[673:end,:,:]

#X_test = X[end-671:end,:,:]
#Y_test = Y[end-671:end,:,:]
#IL TEST
#ultimi 672 della prediction e di Y NORMALE calcola loss on test 

#data = DataLoader((X, Y) ; batchsize = 1)


dimensions = [1, 4 , 4 , 4];#, 16, 32];


machine = RecurMachine(dimensions, sigmoid; pad = 24*4, timeblock = 4*24*2)

model_PM_adam = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

params_PM_adam = Flux.params(model_PM_adam);

opt = ADAM(0.01);

#best_paras = best_parameters( "LBFGS",model,2, X, Y, X_test,Y_test)

loss(x, y) = Flux.Losses.mse(model_PM_adam(x), y)


epochs = Int64[]
loss_on_train_PM_adam = Float64[]
loss_on_test_PM_adam = Float64[]
best_params_PM_adam = Float32[]


for epoch in 1:10
    # Train
    Flux.train!(loss, params_PM_adam, train_data_single, opt)

    # Saving loss
    push!(epochs, epoch)
    push!(loss_on_train_PM_adam, loss(X_train, Y_train))
    push!(loss_on_test_PM_adam, loss(X_test, Y_test))
    @show epoch
    @show loss(X_train, Y_train)
    @show loss(X_test, Y_test)
    
  #  Saving the best parameters
    if epoch > 1
        if is_best(loss_on_test_PM_adam[epoch-1], loss_on_test_PM_adam[epoch])
            best_params_PM_adam = params_PM_adam
        end
    end
end


# Extract and add new trained parameters
if isempty(best_params_PM_adam)
    best_params_PM_adam = params_PM_adam
end


Flux.loadparams!(model_PM_adam, best_params_PM_adam);



ŷ_PM_st = model_PM_adam(X_test)[:]
ŷ_PM = (model_PM_adam(X_test)[:].*s1).+m1


# Visualization
plot(epochs, loss_on_train_PM_adam, lab="Training", c=:blue, lw=2, ylims = (0,2));
plot!(epochs, loss_on_test_PM_adam, lab="Test", c=:green, lw=2, ylims = (0,2));
title!("Recurrent parametric machine architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("recurrentPM_loss_adam.png");





#plotting y predicted vs y true standaridized
plot( y_st , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_PM_st ,alpha = 0.4, lab= "ŷ PM", lw=2) 
plot!( ŷ_CNN_st , alpha = 0.4, lab= "ŷ CNN", lw=2)
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
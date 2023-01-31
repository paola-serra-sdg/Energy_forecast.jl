using ParametricMachinesDemos

X = st[1:end-672,:,:]
Y = st[673:end,:,:]


X_test = X[end-671:end,:,:]
Y_test = Y[end-671:end,:,:]
#IL TEST
#ultimi 672 della prediction e di Y NORMALE calcola loss on test 

data = DataLoader((X, Y) ; batchsize = 1)


dimensions = [1, 2, 4 , 8];#, 16, 32];

machine = RecurMachine(dimensions, sigmoid; pad = 24*4, timeblock = 4*24*2)

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

opt = ADAM(0.1);

params = Flux.params(model);

# Loss function
loss(x, y) = Flux.Losses.mse(model(x), y);

epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc = Float64[]
best_params = Float32[]


for epoch in 1:100
    # Train
    Flux.train!(loss, params, data, opt)

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
            best_params = params
        end
    end
end


# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);

ŷ_PM = (model(X_test)[:].*st).+m

# Visualization
plot(epochs, loss_on_train, lab="Training", c=:blue, lw=2, ylims = (0,5));
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2, ylims = (0,5));
title!("Recurrent parametric machine architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("recurrentPM_loss.png");




#plotting y predicted vs y true
plot( y ,  lab= "y",lw=2)
plot!( ŷ_PM , lab= "ŷ PM", lw=2) 
plot!( ŷ_RNN , lab= "ŷ RNN", lw=2)
title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast.png");

maximum(ŷ_RNN )
minimum(ŷ_RNN)

maximum(ŷ_PM )
minimum(ŷ_PM)
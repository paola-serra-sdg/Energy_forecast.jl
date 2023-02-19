#------CNN   with ADAM optimizer-------------
using Random
N = 24 * 4

Random.seed!(25)
model_CNN_adam = Chain(Conv((N,),1 => 4, pad = (N-1,0),sigmoid),
            Conv((N,),4 => 4, pad = (N-1,0),sigmoid),
            Conv((N,),4 => 4, pad = (N-1,0),sigmoid),
            Conv((N,),4 => 1, pad = (N-1,0)))     |> f64;
            #Dense(32925, 672)) 

            


optimiser = ADAM(0.05);
#0.05 ok
params_CNN_adam = Flux.params(model_CNN_adam);

loss(x, y) = Flux.Losses.mae(model_CNN_adam(x), y)


epochs = Int64[]
loss_on_train_adam = Float64[]
loss_on_test_adam = Float64[]
best_params_CNN_adam = Float32[]


for epoch in 1:100
    Flux.train!(loss, params_CNN_adam, train_data_single, optimiser)
    push!(epochs, epoch)
    push!(loss_on_train_adam, loss(X_train, Y_train))
    push!(loss_on_test_adam,  loss(X_test, Y_test))

    @show epoch
    @show loss(X_train, Y_train)
    @show loss(X_test, Y_test)

    if epoch > 1
        if is_best(loss_on_test_adam[epoch-1], loss_on_test_adam[epoch])
            best_params_CNN_adam = params_CNN_adam
        end
    end
end

# Extract and add new trained parameters
if isempty(best_params_CNN_adam)
    best_params_CNN_adam = params_CNN_adam
end

Flux.loadparams!(model_CNN_adam, best_params_CNN_adam);

ŷ_CNN = (model_CNN_adam(X_test)[:].*s1).+m1
ŷ_CNN_st = model_CNN_adam(X_test)[:]



# Visualization
plot(epochs, loss_on_train_adam, lab="Training", c=:blue, lw=2);
plot!(epochs, loss_on_test_adam, lab="Test", c=:red, lw=2);
title!("Convolutional architecture with ADAM optimizer");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("convolutional_loss_ADAM.pdf");

plot( y , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_CNN ,alpha = 0.4, lab= "ŷ CNN", lw=2) 
title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast_Conv1.pdf");
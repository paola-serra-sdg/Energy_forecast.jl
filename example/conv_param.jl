#------CNN   with ADAM optimizer-------------

N = 24 * 4
Random.seed!(25)
model_CNN_param_adj = Chain(Conv((N,),1 => 8, pad = (N-1,0),sigmoid),
            Conv((N,),8 => 15, pad = (N-1,0),sigmoid),
            Conv((N,),15 => 3, pad = (N-1,0),sigmoid),
            Conv((N,),3 => 1, pad = (N-1,0)))    #|> f64;
            #Dense(32925, 672)) 

            


optimiser = ADAM(0.05);
#0.05 ok
params_CNN_param_adj = Flux.params(model_CNN_param_adj);

loss(x, y) = Flux.Losses.mae(model_CNN_param_adj(x), y)


epochs = Int64[]
loss_on_train_CNN_param_adj = Float64[]
loss_on_test_CNN_param_adj = Float64[]
best_params_CNN_param_adj= Float32[]


for epoch in 1:10
    Flux.train!(loss, params_CNN_param_adj, train_data_single, optimiser)
    push!(epochs, epoch)
    push!(loss_on_train_CNN_param_adj, loss(X_train, Y_train))
    push!(loss_on_test_CNN_param_adj,  loss(X_test, Y_test))

    @show epoch
    @show loss(X_train, Y_train)
    @show loss(X_test, Y_test)

    if epoch > 1
        if is_best(loss_on_test_CNN_param_adj[epoch-1], loss_on_test_CNN_param_adj[epoch])
            best_params_CNN_param_adj = params_CNN_param_adj
        end
    end
end

# Extract and add new trained parameters
if isempty(best_params_CNN_param_adj)
    best_params_CNN_param_adj = params_CNN_param_adj
end

Flux.loadparams!(model_CNN_param_adj, best_params_CNN_param_adj);

ŷ_CNN_2 = (model_CNN_param_adj(X_test)[:].*s1).+m1
ŷ_CNN_2_st = model_CNN_param_adj(X_test)[:]



# Visualization
plot(epochs, loss_on_train_CNN_param_adj, lab="Training", c=:blue, lw=2);
plot!(epochs, loss_on_test_CNN_param_adj, lab="Test", c=:red, lw=2);
title!("Convolutional architecture with ADAM optimizer");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("convolutional2_loss_ADAM");



plot( y , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_CNN_2 ,alpha = 0.4, lab= "ŷ CNN_2", lw=2) 
title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast_cONV2.pdf");
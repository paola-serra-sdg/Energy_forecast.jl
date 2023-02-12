#splitting train and test
X = st[1:end-672,:,:]
Y = st[673:end,:,:]


X_test = X[end-671:end,:,:]
Y_test = Y[end-671:end,:,:]



train_data = DataLoader((X, Y) ; batchsize = 1)
test_data = DataLoader((X_test, Y_test); batchsize = 1)


#------RNN-------------
#model_RNN = Chain(Dense(32925, 4),Dense(4, 1,relu), Dense(1, 672)) |> f64;

N = 24 * 4
model_CNN = Chain(Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
            Conv((N,),1 => 1, pad = (N-1,0)),
            Conv((N,),1 => 1, pad = (N-1,0)),
            Conv((N,),1 => 1, pad = (N-1,0)),
            Conv((N,),1 => 1, pad = (N-1,0)),
            Conv((N,),1 => 1, pad = (N-1,0)))    |> f64;
            #Dense(32925, 672)) 

            


optimiser = ADAM(0.01);

params_CNN = Flux.params(model_CNN);



loss(x, y) = Flux.Losses.mse(model_CNN(x), y)

epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
best_params_CNN = Float32[]


for epoch in 1:200
    Flux.train!(loss, params_CNN, train_data, optimiser)
    push!(epochs, epoch)
    push!(loss_on_train, loss(X, Y))
    push!(loss_on_test,  loss(X_test, Y_test))

    @show epoch
    @show loss(X, Y)
    @show loss(X_test, Y_test)

    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
            best_params_CNN = params_CNN
        end
    end
end

# Extract and add new trained parameters
if isempty(best_params_CNN)
    best_params_CNN = params_CNN
end

Flux.loadparams!(model_CNN, best_params_CNN);

ŷ_CNN = (model_CNN(X_test)[:].*s1).+m1
ŷ_RNN_st = model_CNN(X_test)[:]
# Visualization
plot(epochs, loss_on_train, lab="Training", c=:blue, lw=2);
plot!(epochs, loss_on_test, lab="Test", c=:red, lw=2);
title!("Recurrent architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("recurrent_loss");

maximum(model_RNN(x_test))
minimum(model_RNN(x_test))

maximum(model(X_test))
minimum(model(X_test))

maximum(X_test)
minimum(X_test)
#splitting train and test
x_train = train[1:end-672,:,:]
y_train = train[end+1-672:end,:,:]
x_test = test[1:end-672,:,:]
y_test = test[end+1-672:end,:,:]



train_data = DataLoader((x_train, y_train) ; batchsize = 1)
test_data = DataLoader((x_test, y_test); batchsize = 1)


#------RNN-------------
model = Chain(RNN(32925, 256),Dense(256, 4,relu), Dense(4, 672,relu)) |> f64;

optimiser = ADAM(0.01);

params = Flux.params(model);

loss(x, y) = Flux.Losses.mse(model(x), y)

epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
best_params = Float32[]


for epoch in 1:100
    Flux.train!(loss, params, train_data, optimiser)
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))

    @show epoch
    @show loss(x_train, y_train)
    @show loss(x_test, y_test)

    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
            best_params = params
        end
    end
end

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);

# Visualization
plot(epochs, loss_on_train, lab="Training", c=:blue, lw=2);
plot!(epochs, loss_on_test, lab="Test", c=:red, lw=2);
title!("Recurrent architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("recurrent_loss");
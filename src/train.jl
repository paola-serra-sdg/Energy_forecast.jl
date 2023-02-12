function best_parameters(method::String,model, n_epochs::Int64, x_train, y_train, x_test,y_test)
    if method == "ADAM"
        opt = ADAM(0.1)
        epochs = Int64[]
        loss_on_train = Float64[]
        loss_on_test = Float64[]
        best_params = Float32[]
        loss(x, y) = Flux.Losses.mse(model(x), y)
       
        for epoch in 1:n_epochs
            # Train
            Flux.train!(loss, params, data,opt)
        
            # Saving loss
            push!(epochs, epoch)
            push!(loss_on_train, loss(x_train, y_train))
            push!(loss_on_test,  loss(x_test, y_test))
            @show epoch
            @show loss(x_train, y_train )
            @show loss(x_test, y_test)
        
            # Saving the best parameters
            if epoch > 1
                if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
                #if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
                    best_params = params
                end
            end
        
        end
        return best_params
    else method == "LBFGS"
       # loss() = Flux.Losses.mse(model(x_train), y_train)
        Zygote.refresh()
        lossfun, gradfun, fg!, p0 = optfuns(loss, params)
        res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=10, store_trace=true))
        best_params_PM = res.minimizer
        copy!(params, best_params_PM)
        return params
    end
end 


@assert issorted(energy_dataset.DATE[user_idxs])
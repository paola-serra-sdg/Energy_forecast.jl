function is_best(old_loss, new_loss)
    return old_loss > new_loss
end



#USE MEAN ABSOLUTE ERROR 
#train
error_train_PM  = Flux.Losses.mae(model(X) , test )
error_test_RNN = Flux.Losses.mae(model_RNN(x_train) , y_train)
#test
error_test_PM  = Flux.Losses.mae( ŷ_PM , y )
error_test_RNN = Flux.Losses.mae( ŷ_RNN , y )



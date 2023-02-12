function is_best(old_loss, new_loss)
    return old_loss > new_loss
end



#USE MEAN ABSOLUTE ERROR 

#train
error_train_PM  = Flux.Losses.mae(model(X) , test )
error_test_RNN = Flux.Losses.mae(model_RNN(x_train) , y_train)
#test
error_test_PM  = Flux.Losses.mae( ŷ_PM_st , y )
error_test_RNN = Flux.Losses.mae( ŷ_RNN_st , y )

error_test_PM = Array{Any}(undef, numfiles)
error_test_CNN = Array{Any}(undef, numfiles)
#error test with one user trained
for i in 1:numfiles
    error_test_PM[i] = Flux.Losses.mae( ŷ_PM[i] , y[i] )
    error_test_CNN[i] = Flux.Losses.mae( ŷ_CNN[i] , y[i] )
end

#plotting error_test (the first one is the one used for training)
plot( error_test_PM , alpha = 0.4,  lab= "error_test_PM", lw=2)
plot!( error_test_CNN ,alpha = 0.4, lab= "error_test_CNN", lw=2) 
title!("Error on test PM vs CNN");
yaxis!("error");
xaxis!("user");
savefig("error_test_1user.png");

maximum(error_test_CNN)
maximum(error_test_PM)
minimum(error_test_CNN)
minimum(error_test_PM)





#DEVI SALVARE I MODELLI UNO PER USER E POI CALCOLARE DI NUOVO IL TEST 

# for other user
error_test_PM  = Flux.Losses.mae( y_h_p , y_2 )
df_st[2]

test_2 = df_st[2][end-671:end,:, :]
y_2 = df_st[2][end-671:end]

ŷ_PM_st_2 = model(test_2)[:]

y_h_p =  model(test_2)[:]

mape(y, yhat) = (1/size(y,1))*sum(abs.((y .- yhat) ./ y))
mape(y, ŷ_PM_st)
mape(y_2,ŷ_PM_st_2)


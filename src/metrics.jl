function is_best(old_loss, new_loss)
    return old_loss > new_loss
end



#USE MEAN ABSOLUTE ERROR 

#train
error_train_PM  = Flux.Losses.mae(model_PM_adam(X_train) , Y_train)
error_test_CNN_LBFGS = Flux.Losses.mae(model_CNN(X_test) , Y_test)
#test
error_test_PM_adam  = Flux.Losses.mae( ŷ_PM_st , y_st )
error_test_CNN_adam = Flux.Losses.mae( ŷ_CNN_st , y_st )

error_test_PM = Array{Any}(undef, numfiles)
error_test_CNN = Array{Any}(undef, numfiles)
#error test with one user trained
for i in 1:numfiles
    #error_test_PM[i] = Flux.Losses.mae( ŷ_PM_multi[i] , y_multi[i] )
    #error_test_CNN[i] = Flux.Losses.mae( model_CNN_adam(X_) , model_CNN_adam[i] )
    error_test_CNN[i] = Flux.Losses.mae( model_CNN_adam(X_) , y )
end

#plotting error_test (the first one is the one used for training)
bar( x = user[1:20], error_test_PM[1:20] , alpha = 0.4,  lab= "error_test_PM", lw=2 ,xrotation=45 )
bar!(x= user[1:20], error_test_CNN[1:20] ,alpha = 0.4, lab= "error_test_CNN", lw=2, xrotation=45 ) 
title!("Error on test PM vs CNN");
yaxis!("error");
xaxis!("user");
savefig("error_test_1user.png");

plot( error_test_PM , alpha = 0.4,  lab= "error_test_PM", lw=2)
plot!( error_test_CNN ,alpha = 0.4, lab= "error_test_CNN", lw=2) 
title!("Error on test PM vs CNN");
yaxis!("error");
xaxis!("user");
savefig("error_test_1user.png");


p = Vector{Any}(undef, numfiles)
for i in 1:2
    p[i] = plot( y_test[i][:], alpha = 0.4,  lab= y ,lw=2)
    #plot( , alpha = 0.4,  lab= y ,lw=2)
    title!(string("Predicted vs true - ",user[i]));
    display(p[i])
    sleep(1)
    savefig(string(user[i],"_adam.pdf"))
end








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


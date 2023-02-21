function is_best(old_loss, new_loss)
    return old_loss > new_loss
end


#ADAM
# MSE test error on user trained
error_test_PM_adam  = Flux.Losses.mse( ŷ_PM_st , y_st )
error_test_CNN_adam= Flux.Losses.mse( ŷ_CNN_st , y_st )
error_test_CNN2_adam = Flux.Losses.mse( ŷ_CNN_2_st , y_st )

#MAE 
error_test_PM_adam_mae  = Flux.Losses.mae( ŷ_PM_st , y_st )
error_test_CNN_adam_mae= Flux.Losses.mae( ŷ_CNN_st , y_st )
error_test_CNN2_adam_mae = Flux.Losses.mae( ŷ_CNN_2_st , y_st )






# MSE test error on user trained
error_test_PM_lbfgs  = Flux.Losses.mse( ŷ_PM_lbfgs_st , y_st )
error_test_CNN_lbfgs= Flux.Losses.mse( ŷ_CNN_lbfgs_st , y_st )
error_test_CNN2_lbfgs = Flux.Losses.mse( ŷ_CNN2_lbfgs_st , y_st )

#MAE 
error_test_PM_lbfgs_mae  = Flux.Losses.mae( ŷ_PM_lbfgs_st , y_st )
error_test_CNN_lbfgs_mae= Flux.Losses.mae( ŷ_CNN_lbfgs_st , y_st )
error_test_CNN2_lbfgs_mae = Flux.Losses.mae( ŷ_CNN2_lbfgs_st , y_st )







#MSE DISTRIBUTION

MSE_test_PM = Array{Any}(undef, numfiles)
MSE_test_CNN = Array{Any}(undef, numfiles)
MSE_test_CNN_2 = Array{Any}(undef, numfiles)
#error test with one user trained
#ADAM
for i in 1:numfiles
    MSE_test_PM[i] = Flux.Losses.mse( ŷ_PM_multi_st[i] , y_multi_st[i])
    MSE_test_CNN[i] = Flux.Losses.mse( ŷ_CNN_multi_st[i] , y_multi_st[i])
    MSE_test_CNN_2[i] = Flux.Losses.mse( ŷ_CNN2_multi_st[i]  , y_multi_st[i] )
end



Plots.histogram(MSE_test_PM, alpha = 0.4,label="PM", normalize=:pdf, color=:green)
Plots.histogram!(MSE_test_CNN,alpha = 0.4, label="CNN", normalize=:pdf, color=:pink)
Plots.histogram!(MSE_test_CNN_2, alpha = 0.4,label="CNN2", normalize=:pdf, color=:blue)
savefig("Mse_adam.pdf")


#MSE L-BFGS
MSE_test_PM_lbfgs = Array{Any}(undef, numfiles)
MSE_test_CNN_lbfgs = Array{Any}(undef, numfiles)
MSE_test_CNN_2_lbfgs = Array{Any}(undef, numfiles)
#error test with one user trained
#ADAM
for i in 1:numfiles
    MSE_test_PM_lbfgs[i] = Flux.Losses.mse( ŷ_PM_multi_lbfgs_st[i], y_multi_st[i])
    MSE_test_CNN_lbfgs[i] = Flux.Losses.mse(  ŷ_CNN_multi_lbfgs_st[i] , y_multi_st[i])
    MSE_test_CNN_2_lbfgs[i] = Flux.Losses.mse( ŷ_CNN2_multi_lbfgs_st[i]  , y_multi_st[i] )
end

Plots.histogram(MSE_test_PM_lbfgs,legend=false, alpha = 0.4,bins=20, normalize=:pdf, color=:green)
Plots.histogram!(MSE_test_CNN_lbfgs, legend=false, alpha = 0.4,bins=:sqrt,normalize=:pdf, color=:pink)
Plots.histogram!(MSE_test_CNN_2_lbfgs,legend=false, alpha = 0.4,  bins=20, normalize=:pdf, color=:blue)
savefig("MSE_lbfgs.pdf")

###MAE
MAE_test_PM = Array{Any}(undef, numfiles)
MAE_test_CNN = Array{Any}(undef, numfiles)
MAE_test_CNN_2 = Array{Any}(undef, numfiles)
#error test with one user trained
#ADAM
for i in 1:numfiles
    MAE_test_PM[i] = Flux.Losses.mae( ŷ_PM_multi_st[i] , y_multi_st[i])
    MAE_test_CNN[i] = Flux.Losses.mae( ŷ_CNN_multi_st[i] , y_multi_st[i])
    MAE_test_CNN_2[i] = Flux.Losses.mae( ŷ_CNN2_multi_st[i]  , y_multi_st[i] )
end

MAE_ADAM = Pandas.DataFrame(MAE_test_CNN)

Plots.histogram(MAE_test_PM,legend=false, alpha = 0.4,normalize=:pdf, color=:green)
Plots.histogram!(MAE_test_CNN,legend=false, alpha = 0.4, normalize=:pdf, color=:pink)
Plots.histogram!(MAE_test_CNN_2, legend=false, alpha = 0.4, normalize=:pdf, color=:blue)
savefig("Mae_adam.pdf")


#MAE L-BFGS
MAE_test_PM_lbfgs = Array{Any}(undef, numfiles)
MAE_test_CNN_lbfgs = Array{Any}(undef, numfiles)
MAE_test_CNN_2_lbfgs = Array{Any}(undef, numfiles)
#error test with one user trained
#ADAM
for i in 1:numfiles
    MAE_test_PM_lbfgs[i] = Flux.Losses.mae( ŷ_PM_multi_lbfgs_st[i], y_multi_st[i])
    MAE_test_CNN_lbfgs[i] = Flux.Losses.mae(  ŷ_CNN_multi_lbfgs_st[i] , y_multi_st[i])
    MAE_test_CNN_2_lbfgs[i] = Flux.Losses.mae( ŷ_CNN2_multi_lbfgs_st[i]  , y_multi_st[i] )
end

Plots.histogram(MAE_test_PM_lbfgs,legend=false, alpha = 0.4,bins=:sqrt, normalize=:pdf, color=:green)
Plots.histogram!(MAE_test_CNN_lbfgs, legend=false, alpha = 0.4, bins=:sqrt,normalize=:pdf, color=:pink)
Plots.histogram!(MAE_test_CNN_2_lbfgs, legend=false, alpha = 0.4, bins=:sqrt, normalize=:pdf, color=:blue)
savefig("Mae_lbfgs.pdf")












#plotting forecast(the first one is the one used for training)



p = Vector{Any}(undef, numfiles)
for i in 1:10
    p[i] = plot( y_test[i][:], alpha = 0.4,  lab= y ,lw=2)
    plot!( model_PM_adam(x_test[i])[:],alpha = 0.4, lab= "ŷ PM", lw=2) 
    plot!( model_CNN_adam(x_test[i])[:], alpha = 0.4, lab= "ŷ CNN", lw=2)
    plot!( model_CNN_param_adj(x_test[i])[:] , alpha = 0.4, lab= "ŷ CNN", lw=2)
    title!(string("Predicted vs true - ",user[i]));
    display(p[i])
    sleep(1)
    savefig(string(user[i],"_adam.pdf"))
end


p_lbfgs = Vector{Any}(undef, numfiles)
for i in 1:10
    p_lbfgs[i] = plot( y_test[i][:], alpha = 0.4,  lab= y ,lw=2)
    plot!( model_PM_lbfgs(X_test[i])[:] ,alpha = 0.4, lab= "ŷ PM", lw=2) 
    plot!( model_CNN_adam(x_test[i])[:], alpha = 0.4, lab= "ŷ CNN", lw=2)
    plot!( ŷ_CNN_2 , alpha = 0.4, lab= "ŷ CNN", lw=2)
    title!(string("Predicted vs true - ",user[i]));
    display(p[i])
    sleep(1)
    savefig(string(user[i],"_lbfgs.pdf"))
end





maximum(error_test_CNN)
maximum(error_test_PM)
minimum(error_test_CNN)
minimum(error_test_PM)








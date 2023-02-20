function is_best(old_loss, new_loss)
    return old_loss > new_loss
end


# MSE test error on user trained
error_test_PM_lbfgs  = Flux.Losses.mse( ŷ_PM_st , y_st )
error_test_CNN_lbfgs= Flux.Losses.mse( ŷ_CNN_lbfgs_st , y_st )
error_test_CNN2_lbfgs = Flux.Losses.mse( ŷ_CNN2_lbfgs_st , y_st )

#MAE 
error_test_PM_lbfgs_mae  = Flux.Losses.mae( ŷ_PM_st , y_st )
error_test_CNN_lbfgs_mae= Flux.Losses.mae( ŷ_CNN_lbfgs_st , y_st )
error_test_CNN2_lbfgs_mae = Flux.Losses.mae( ŷ_CNN2_lbfgs_st , y_st )


#MSE DISTRIBUTION



MAE_test_PM = Array{Any}(undef, numfiles)
MAE_test_CNN = Array{Any}(undef, numfiles)
MAE_test_CNN_2 = Array{Any}(undef, numfiles)
#error test with one user trained
#ADAM
for i in 1:numfiles
    MAE_test_PM[i] = Flux.Losses.mae( ŷ_PM_multi[i] , y_test[i])
    MAE_test_CNN[i] = Flux.Losses.mae( ŷ_CNN_multi[i] , y_test[i] )
    MAE_test_CNN_2[i] = Flux.Losses.mae( ŷ_CNN2_multi[i]  , y_test[i] )
end

MAE_ADAM = Pandas.DataFrame(MAE_test_CNN)

Plots.histogram(MAE_test_PM, label="Experimental", normalize=:pdf, color=:green)
#plot!(p, label="Analytical", lw=3, color=:red)
#xlims!(-5, 5)
#ylims!(0, 0.4)
title!("Normal distribution, 1000 samples")

Plots.histogram!(MAE_test_CNN, label="Experimental", normalize=:pdf, color=:gray)
Plots.histogram!(MAE_test_CNN_2, label="Experimental", normalize=:pdf, color=:blue)

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








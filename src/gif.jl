
import Printf.@sprintf
import Plots:Animation, buildanimation  



##trained user
predictions=[]
push!(predictions,y)
push!(predictions,ŷ_PM)
push!(predictions,ŷ_CNN)
push!(predictions,ŷ_CNN_2)

labels = ["y","ŷ PM","ŷ CNN-1","ŷ CNN-2"]


j = 1
for i in 1:4
    if i== 1
        plot(predictions[i], lab = labels[i],alpha = 0.4, lw=2);
        savefig("visualization/gif/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions[i], lab = labels[i],alpha = 0.4, lw=2);
        savefig("visualization/gif/" * @sprintf("%06d.png", j))
        j +=1
    end
end

#nframes è il numero di immagini che devi inserire nella gif
#visualiztion/... è la cartella dove vuoi salvare la gif e original_data.gif è il nome della gif
#fps è la velocità con cui vuoi che si muova la gif

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif", fnames); 
buildanimation(anim, "visualization/gif/original_data.gif", fps = 0.50, show_msg=false)




###l-BFGS

predictions_lbfgs=[]
push!(predictions_lbfgs,y)
push!(predictions_lbfgs,ŷ_PM_lbfgs)
push!(predictions_lbfgs,ŷ_CNN_lbfgs)
push!(predictions_lbfgs,ŷ_CNN2_lbfgs)


j = 1
for i in 1:4
    if i== 1
        plot(predictions_lbfgs[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions_lbfgs[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/" * @sprintf("%06d.png", j))
        j +=1
    end
end

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/lbfgs/", fnames); 
buildanimation(anim, "visualization/gif/lbfgs/trained_lbfgs.gif", fps = 0.50, show_msg=false)


#examples
predictions_lbfgs=[]
push!(predictions_lbfgs,y)
push!(predictions_lbfgs,ŷ_PM_lbfgs)
push!(predictions_lbfgs,ŷ_CNN_lbfgs)
push!(predictions_lbfgs,ŷ_CNN2_lbfgs)


j = 1
for i in 1:4
    if i== 1
        plot(predictions_lbfgs[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions_lbfgs[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/" * @sprintf("%06d.png", j))
        j +=1
    end
end

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/lbfgs/", fnames); 
buildanimation(anim, "visualization/gif/lbfgs/trained_lbfgs.gif", fps = 0.50, show_msg=false)


#EXAMPLES
predictions_adam_1143954=[]
predictions_lbfgs_1143954=[]


push!(predictions_adam_1143954, y_multi[3][:])
push!(predictions_adam_1143954, ŷ_PM_multi[3])
push!(predictions_adam_1143954, ŷ_CNN_multi[3])
push!(predictions_adam_1143954, ŷ_CNN2_multi[3])

push!(predictions_lbfgs_1143954, y_multi[3][:])
push!(predictions_lbfgs_1143954, ŷ_PM_multi_lbfgs[3])
push!(predictions_lbfgs_1143954, ŷ_CNN_multi_lbfgs[3])
push!(predictions_lbfgs_1143954, ŷ_CNN2_multi_lbfgs[3])




j = 1
for i in 1:4
    if i== 1
        plot(predictions_adam_1143954[i],legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/adam/1143954/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions_adam_1143954[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/adam/1143954/" * @sprintf("%06d.png", j))
        j +=1
    end
end

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/adam/1143954", fnames); 
buildanimation(anim, "visualization/gif/adam/predictions_adam_1143954.gif", fps = 0.50, show_msg=false)


j = 1
for i in 1:4
    if i== 1
        plot(predictions_lbfgs_1143954[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/1143954/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions_lbfgs_1143954[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/1143954/" * @sprintf("%06d.png", j))
        j +=1
    end
end

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/lbfgs/1143954/", fnames); 
buildanimation(anim, "visualization/gif/lbfgs/predictions_lbfgs_1143954.gif", fps = 0.50, show_msg=false)

predictions_adam_1362155=[]
predictions_lbfgs_1362155=[]


push!(predictions_adam_1362155, y_multi[216][:])
push!(predictions_adam_1362155, ŷ_PM_multi[216])
push!(predictions_adam_1362155, ŷ_CNN_multi[216])
push!(predictions_adam_1362155, ŷ_CNN2_multi[216])

push!(predictions_lbfgs_1362155, y_multi[216][:])
push!(predictions_lbfgs_1362155, ŷ_PM_multi_lbfgs[216])
push!(predictions_lbfgs_1362155, ŷ_CNN_multi_lbfgs[216])
push!(predictions_lbfgs_1362155, ŷ_CNN2_multi_lbfgs[216])




j = 1
for i in 1:4
    if i== 1
        plot(predictions_adam_1362155[i],legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/adam/1362155/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions_adam_1362155[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/adam/1362155/" * @sprintf("%06d.png", j))
        j +=1
    end
end

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/adam/1362155", fnames); 
buildanimation(anim, "visualization/gif/adam/predictions_adam_1362155.gif", fps = 0.50, show_msg=false)


j = 1
for i in 1:4
    if i== 1
        plot(predictions_lbfgs_1362155[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/1362155/" * @sprintf("%06d.png", j))
        j +=1
    else 
        plot!(predictions_lbfgs_1362155[i], legend=false,alpha = 0.4, lw=2);
        savefig("visualization/gif/lbfgs/1362155/" * @sprintf("%06d.png", j))
        j +=1
    end
end

nframes = 4         
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/lbfgs/1362155/", fnames); 
buildanimation(anim, "visualization/gif/lbfgs/predictions_lbfgs_1362155.gif", fps = 0.50, show_msg=false)



### GENERALIZATION


##Mse_adam

j = 1
for i in 1:3
    if i== 1
        Plots.histogram(MSE_test_PM, alpha = 0.4,label="PM", normalize=:pdf, color=:green)
        savefig("visualization/gif/adam/MSE/" * @sprintf("%06d.png", j))
        j +=1
    elseif i==2
        Plots.histogram!(MSE_test_CNN,alpha = 0.4, label="CNN-1", normalize=:pdf, color=:pink)

        savefig("visualization/gif/adam/MSE/" * @sprintf("%06d.png", j))
        j +=1
    else
        Plots.histogram!(MSE_test_CNN_2, alpha = 0.4,label="CNN-2", normalize=:pdf, color=:blue)
        savefig("visualization/gif/adam/MSE/" * @sprintf("%06d.png", j))
    end
end

nframes = 3  
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/adam/MSE/", fnames); 
buildanimation(anim, "visualization/gif/adam/MSE/MSE_ADAM.gif", fps = 0.50, show_msg=false)


############ MSE_lbfgs

j = 1
for i in 1:3
    if i== 1
        Plots.histogram(MSE_test_PM_lbfgs,legend=false, alpha = 0.4,bins=20, normalize=:pdf, color=:green)
        savefig("visualization/gif/lbfgs/MSE/" * @sprintf("%06d.png", j))
        j +=1
    elseif i==2
        Plots.histogram!(MSE_test_CNN_lbfgs, legend=false, alpha = 0.4,bins=:sqrt,normalize=:pdf, color=:pink)
        savefig("visualization/gif/lbfgs/MSE/" * @sprintf("%06d.png", j))
        j +=1
    else
        Plots.histogram!(MSE_test_CNN_2_lbfgs,legend=false, alpha = 0.4,  bins=20, normalize=:pdf, color=:blue)
        savefig("visualization/gif/lbfgs/MSE/" * @sprintf("%06d.png", j))
    end
end

nframes = 3  
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/lbfgs/MSE/", fnames); 
buildanimation(anim, "visualization/gif/lbfgs/MSE/MSE_lbfgs.gif", fps = 0.50, show_msg=false)



#####MAE

## ADAM


j = 1
for i in 1:3
    if i== 1
        Plots.histogram(MAE_test_PM,legend=false, alpha = 0.4,normalize=:pdf, color=:green)
        savefig("visualization/gif/adam/MAE/" * @sprintf("%06d.png", j))
        j +=1
    elseif i==2
        Plots.histogram!(MAE_test_CNN,legend=false, alpha = 0.4, normalize=:pdf, color=:pink)
        savefig("visualization/gif/adam/MAE/" * @sprintf("%06d.png", j))
        j +=1
    else
        Plots.histogram!(MAE_test_CNN_2, legend=false, alpha = 0.4, normalize=:pdf, color=:blue)
        savefig("visualization/gif/adam/MAE/" * @sprintf("%06d.png", j))
    end
end

nframes = 3  
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/adam/MAE/", fnames); 
buildanimation(anim, "visualization/gif/adam/MAE/MAE_ADAM.gif", fps = 0.50, show_msg=false)



###MAE LBFGS

j = 1
for i in 1:3
    if i== 1
        Plots.histogram(MAE_test_PM_lbfgs,legend=false, alpha = 0.4,bins=:sqrt, normalize=:pdf, color=:green)
        savefig("visualization/gif/lbfgs/MAE/" * @sprintf("%06d.png", j))
        j +=1
    elseif i==2
        Plots.histogram!(MAE_test_CNN_lbfgs, legend=false, alpha = 0.4, bins=:sqrt,normalize=:pdf, color=:pink)
        savefig("visualization/gif/lbfgs/MAE/" * @sprintf("%06d.png", j))
        j +=1
    else
        Plots.histogram!(MAE_test_CNN_2_lbfgs, legend=false, alpha = 0.4, bins=:sqrt, normalize=:pdf, color=:blue)
        savefig("visualization/gif/lbfgs/MAE/" * @sprintf("%06d.png", j))
    end
end

nframes = 3  
fnames = [@sprintf("%06d.png", k) for k  in 1:nframes]   
anim = Animation("visualization/gif/lbfgs/MAE/", fnames); 
buildanimation(anim, "visualization/gif/lbfgs/MAE/MAE_lbfgs.gif", fps = 0.50, show_msg=false)
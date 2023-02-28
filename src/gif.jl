
import Printf.@sprintf
import Plots:Animation, buildanimation  



plot(epochs, loss_on_train_PM_adam, lab="PM", c=:green, alpha = 0.4, lw=2, ylims = (0,6));
plot!(epochs, loss_on_train_adam, lab="CNN-1", c=:red, alpha = 0.4,lw=2, ylims = (0,6));
plot!(epochs, loss_on_train_CNN_param_adj, lab="CNN-2", c=:blue, alpha = 0.4,lw=2, ylims = (0,6));
#title!("Convolutional architecture with ADAM optimizer");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("loss_ADAM.pdf");

losses = []
push!(losses, loss_on_train_PM_adam)
push!(losses, loss_on_train_adam)
push!(losses,loss_on_train_CNN_param_adj)


#plotting y predicted vs y true
plot( y , alpha = 0.4,  lab= "y",lw=2)
plot!( ŷ_PM ,alpha = 0.4, lab= "ŷ PM", lw=2) 
plot!( ŷ_CNN, alpha = 0.4, lab= "ŷ CNN-1", lw=2)
plot!( ŷ_CNN_2 , alpha = 0.4, lab= "ŷ CNN-2", lw=2)
#title!("Predicted vs True");
yaxis!("Energy demand");
xaxis!("Time");
savefig("energy_forecast.pdf");

predictions=[]
push!(predictions,y)
push!(predictions,ŷ_PM)
push!(predictions,ŷ_CNN)
push!(predictions,ŷ_CNN_2)

labels = ["y","ŷ PM","ŷ CNN-1","ŷ CNN-2"]

plot( predictions[1], lab = labels[1],c=:green ,alpha = 0.4, lw=2)

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
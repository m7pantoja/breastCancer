def train(model,train_loader,criterion,optimizer,num_epochs,device,task):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for images, labels in train_loader:

            images = images.to(device)

            if task == 'binary':
                labels = labels.float().unsqueeze(1).to(device)
            elif task == 'multiclass':
                labels = labels.long().to(device)
            else:
                raise ValueError("Argumento inválido para el parámetro task. Debe ser 'binary' o 'multiclass'")

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward() # backpropagation 
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} — Train Loss: {epoch_loss:.4f}")
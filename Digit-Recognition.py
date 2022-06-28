#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import torch
import torchvision
from torchvision.datasets import MNIST


# In[2]:


# Download training dataset
dataset = MNIST(root='data/', download=True)


# When this statement is executed for the first time, it downloads the data to the `data/` directory next to the notebook and creates a PyTorch `Dataset`. On subsequent executions, the download is skipped as the data is already downloaded. Let's check the size of the dataset.

# In[3]:


len(dataset)


# The dataset has 60,000 images that we'll use to train the model. There is also an additional test set of 10,000 images used for evaluating models and reporting metrics in papers and reports. We can create the test dataset using the `MNIST` class by passing `train=False` to the constructor.

# In[4]:


test_dataset = MNIST(root='data/', train=False)
len(test_dataset)


# In[5]:


dataset[0]


# It's a pair, consisting of a 28x28px image and a label. The image is an object of the class `PIL.Image.Image`, which is a part of the Python imaging library [Pillow](https://pillow.readthedocs.io/en/stable/). We can view the image within Jupyter using [`matplotlib`](https://matplotlib.org/), the de-facto plotting and graphing library for data science in Python.

# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label:', label)


# In[8]:


image, label = dataset[10]
plt.imshow(image, cmap='gray')
print('Label:', label)


# In[9]:


import torchvision.transforms as transforms


# In[10]:


# MNIST dataset (images and labels)
dataset = MNIST(root='data/', 
                train=True,
                transform=transforms.ToTensor())


# In[11]:


img_tensor, label = dataset[0]
print(img_tensor.shape, label)


# The image is now converted to a 1x28x28 tensor. The first dimension tracks color channels. The second and third dimensions represent pixels along the height and width of the image, respectively. Since images in the MNIST dataset are grayscale, there's just one channel. Other datasets have images with color, in which case there are three channels: red, green, and blue (RGB). 

# In[12]:


print(img_tensor[0,10:15,10:15])
print(torch.max(img_tensor), torch.min(img_tensor))


# The values range from 0 to 1, with `0` representing black, `1` white, and the values in between different shades of grey. We can also plot the tensor as an image using `plt.imshow`.

# In[13]:


# Plot the image by passing in the 28x28 matrix
plt.imshow(img_tensor[0,10:15,10:15], cmap='gray');


# Note that we need to pass just the 28x28 matrix to `plt.imshow`, without a channel dimension. We also pass a color map (`cmap=gray`) to indicate that we want to see a grayscale image.

# ## Training and Validation Datasets
# 
# While building real-world machine learning models, it is quite common to split the dataset into three parts:
# 
# 1. **Training set** - used to train the model, i.e., compute the loss and adjust the model's weights using gradient descent.
# 2. **Validation set** - used to evaluate the model during training, adjust hyperparameters (learning rate, etc.), and pick the best version of the model.
# 3. **Test set** - used to compare different models or approaches and report the model's final accuracy.
# 
# In the MNIST dataset, there are 60,000 training images and 10,000 test images. The test set is standardized so that different researchers can report their models' results against the same collection of images. 
# 
# Since there's no predefined validation set, we must manually split the 60,000 images into training and validation datasets. Let's set aside 10,000 randomly chosen images for validation. We can do this using the `random_spilt` method from PyTorch.

# In[14]:


from torch.utils.data import random_split

train_ds, val_ds = random_split(dataset, [50000, 10000])
len(train_ds), len(val_ds)


# It's essential to choose a random sample for creating a validation set. Training data is often sorted by the target labels, i.e., images of 0s, followed by 1s, followed by 2s, etc. If we create a validation set using the last 20% of images, it would only consist of 8s and 9s. In contrast, the training set would contain no 8s or 9s. Such a training-validation would make it impossible to train a useful model.
# 
# We can now create data loaders to help us load the data in batches. We'll use a batch size of 128.
# 
# 
# 

# In[15]:


from torch.utils.data import DataLoader

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


# We set `shuffle=True` for the training data loader to ensure that the batches generated in each epoch are different. This randomization helps generalize & speed up the training process. On the other hand, since the validation data loader is used only for evaluating the model, there is no need to shuffle the images. 
# 

# ## Model
# 
# Now that we have prepared our data loaders, we can define our model.
# 
# * A **logistic regression** model is almost identical to a linear regression model. It contains weights and bias matrices, and the output is obtained using simple matrix operations (`pred = x @ w.t() + b`). 
# 
# * Since `nn.Linear` expects each training example to be a vector, each `1x28x28` image tensor is _flattened_ into a vector of size 784 `(28*28)` before being passed into the model. 
# 
# * The output for each image is a vector of size 10, with each element signifying the probability of a particular target label (i.e., 0 to 9). The predicted label for an image is simply the one with the highest probability.

# In[16]:


import torch.nn as nn

input_size = 28*28
num_classes = 10

# Logistic regression model
model = nn.Linear(input_size, num_classes)


# Of course, this model is a lot larger than our previous model in terms of the number of parameters. Let's take a look at the weights and biases.

# In[17]:


print(model.weight.shape)
model.weight


# In[18]:


print(model.bias.shape)
model.bias


# Although there are a total of 7850 parameters here, conceptually, nothing has changed so far. Let's try and generate some outputs using our model. We'll take the first batch of 100 images from our dataset and pass them into our model.

# In[19]:


for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)
    print(outputs)
    break


# In[20]:


images.shape


# In[21]:


images.reshape(128, 784).shape


# The code above leads to an error because our input data does not have the right shape. Our images are of the shape 1x28x28, but we need them to be vectors of size 784, i.e., we need to flatten them. We'll use the `.reshape` method of a tensor, which will allow us to efficiently 'view' each image as a flat vector without really creating a copy of the underlying data. To include this additional functionality within our model, we need to define a custom model by extending the `nn.Module` class from PyTorch. 

# In[22]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()


# Inside the `__init__` constructor method, we instantiate the weights and biases using `nn.Linear`. And inside the `forward` method, which is invoked when we pass a batch of inputs to the model, we flatten the input tensor and pass it into `self.linear`.
# 
# `xb.reshape(-1, 28*28)` indicates to PyTorch that we want a *view* of the `xb` tensor with two dimensions. The length along the 2nd dimension is 28\*28 (i.e., 784). One argument to `.reshape` can be set to `-1` (in this case, the first dimension) to let PyTorch figure it out automatically based on the shape of the original tensor.
# 
# Note that the model no longer has `.weight` and `.bias` attributes (as they are now inside the `.linear` attribute), but it does have a `.parameters` method that returns a list containing the weights and bias.

# In[23]:


model.linear


# In[24]:


print(model.linear.weight.shape, model.linear.bias.shape)
list(model.parameters())


# We can use our new custom model in the same way as before. Let's see if it works.

# In[25]:


for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

print('outputs.shape : ', outputs.shape)
print('Sample outputs :\n', outputs[:2].data)


# For each of the 100 input images, we get 10 outputs, one for each class. As discussed earlier, we'd like these outputs to represent probabilities. Each output row's elements must lie between 0 to 1 and add up to 1, which is not the case. 
# 
# To convert the output rows into probabilities, we use the softmax function, which has the following formula:
# 
# ![softmax](./assets/softmax.png)
# 
# First, we replace each element ${\large{y_i}}$ in an output row by ${\large{e^{y_i}}}$, making all the elements positive. 
# 
# ![](./assets/graph.png)
# 
# 
# Then, we divide them by their sum to ensure that they add up to 1. The resulting vector can thus be interpreted as probabilities.

# In[26]:


import torch.nn.functional as F


# The softmax function is included in the `torch.nn.functional` package and requires us to specify a dimension along which the function should be applied.

# In[27]:


outputs[:2]


# In[28]:


# Apply softmax for each output row
probs = F.softmax(outputs, dim=1)

# Look at sample probabilities
print("Sample probabilities:\n", probs[:2].data)

# Add up the probabilities of an output row
print("Sum: ", torch.sum(probs[0]).item())


# Finally, we can determine the predicted label for each image by simply choosing the index of the element with the highest probability in each output row. We can do this using `torch.max`, which returns each row's largest element and the corresponding index.

# In[29]:


max_probs, preds = torch.max(probs, dim=1)
print(preds)
print(max_probs)


# The numbers printed above are the predicted labels for the first batch of training images. Let's compare them with the actual labels.

# In[30]:


labels


# Most of the predicted labels are different from the actual labels. That's because we have started with randomly initialized weights and biases. We need to train the model, i.e., adjust the weights using gradient descent to make better predictions.

# ## Evaluation Metric and Loss Function

# Just as with linear regression, we need a way to evaluate how well our model is performing. A natural way to do this would be to find the percentage of labels that were predicted correctly, i.e,. the **accuracy** of the predictions. 

# In[31]:


outputs[:2]


# In[32]:


torch.sum(preds == labels)


# In[33]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# The `==` operator performs an element-wise comparison of two tensors with the same shape and returns a tensor of the same shape, containing `True` for unequal elements and `False` for equal elements. Passing the result to `torch.sum` returns the number of labels that were predicted correctly. Finally, we divide by the total number of images to get the accuracy. 
# 
# Note that we don't need to apply softmax to the outputs since its results have the same relative order. This is because ${\large{e^{x}}}$ is an increasing function, i.e., if ${\large{y_1 > y_2}}$, then ${\large{e^{y_1} > e^{y_2}}}$. The same holds after averaging out the values to get the softmax.
# 
# Let's calculate the accuracy of the current model on the first batch of data. 

# In[34]:


accuracy(outputs, labels)


# In[35]:


probs


# Accuracy is an excellent way for us (humans) to evaluate the model. However, we can't use it as a loss function for optimizing our model using gradient descent for the following reasons:
# 
# 1. It's not a differentiable function. `torch.max` and `==` are both non-continuous and non-differentiable operations, so we can't use the accuracy for computing gradients w.r.t the weights and biases.
# 
# 2. It doesn't take into account the actual probabilities predicted by the model, so it can't provide sufficient feedback for incremental improvements. 
# 
# For these reasons, accuracy is often used as an **evaluation metric** for classification, but not as a loss function. A commonly used loss function for classification problems is the **cross-entropy**, which has the following formula:
# 
# ![cross-entropy](./assets/entropy.png)
# 
# While it looks complicated, it's actually quite simple:
# 
# * For each output row, pick the predicted probability for the correct label. E.g., if the predicted probabilities for an image are `[0.1, 0.3, 0.2, ...]` and the correct label is `1`, we pick the corresponding element `0.3` and ignore the rest.
# 
# * Then, take the [logarithm](https://en.wikipedia.org/wiki/Logarithm) of the picked probability. If the probability is high, i.e., close to 1, then its logarithm is a very small negative value, close to 0. And if the probability is low (close to 0), then the logarithm is a very large negative value. We also multiply the result by -1, which results is a large postive value of the loss for poor predictions.
# 
# ![](./assets/graph2.png)
# 
# * Finally, take the average of the cross entropy across all the output rows to get the overall loss for a batch of data.
# 
# Unlike accuracy, cross-entropy is a continuous and differentiable function. It also provides useful feedback for incremental improvements in the model (a slightly higher probability for the correct label leads to a lower loss). These two factors make cross-entropy a better choice for the loss function.
# 
# As you might expect, PyTorch provides an efficient and tensor-friendly implementation of cross-entropy as part of the `torch.nn.functional` package. Moreover, it also performs softmax internally, so we can directly pass in the model's outputs without converting them into probabilities.

# In[36]:


outputs


# In[37]:


loss_fn = F.cross_entropy


# In[38]:


# Loss for current batch of data
loss = loss_fn(outputs, labels)
print(loss)


# We know that cross-entropy is the negative logarithm of the predicted probability of the correct label averaged over all training samples. Therefore, one way to interpret the resulting number e.g. `2.23` is look at ${\large{e^{-2.23}}}$ which is around `0.1` as the predicted probability of the correct label, on average. *The lower the loss, The better the model.*

# ## Training the model
# 
# Now that we have defined the data loaders, model, loss function and optimizer, we are ready to train the model. The training process is identical to linear regression, with the addition of a "validation phase" to evaluate the model in each epoch. Here's what it looks like in pseudocode:
# 
# ```
# for epoch in range(num_epochs):
#     # Training phase
#     for batch in train_loader:
#         # Generate predictions
#         # Calculate loss
#         # Compute gradients
#         # Update weights
#         # Reset gradients
#     
#     # Validation phase
#     for batch in val_loader:
#         # Generate predictions
#         # Calculate loss
#         # Calculate metrics (accuracy etc.)
#     # Calculate average validation loss & metrics
#     
#     # Log epoch, loss & metrics for inspection
# ```
# 
# Some parts of the training loop are specific the specific problem we're solving (e.g. loss function, metrics etc.) whereas others are generic and can be applied to any deep learning problem. 
# 
# We'll include the problem-independent parts within a function called `fit`, which will be used to train the model. The problem-specific parts will be implemented by adding new methods to the `nn.Module` class.
# 
# 

# In[39]:


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history


# The `fit` function records the validation loss and metric from each epoch. It returns a history of the training, useful for debugging & visualization.
# 
# Configurations like batch size, learning rate, etc. (called hyperparameters), need to picked in advance while training machine learning models. Choosing the right hyperparameters is critical for training a reasonably accurate model within a reasonable amount of time. It is an active area of research and experimentation in machine learning. Feel free to try different learning rates and see how it affects the training process.
# 
# 
# Let's define the `evaluate` function, used in the validation phase of `fit`.

# In[40]:


l1 = [1, 2, 3, 4, 5]


# In[41]:


l2 = [x*2 for x in l1]
l2


# In[42]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# 
# Finally, let's redefine the `MnistModel` class to include additional methods `training_step`, `validation_step`, `validation_epoch_end`, and `epoch_end` used by `fit` and `evaluate`.

# In[43]:


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss, 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    
model = MnistModel()


# Before we train the model, let's see how the model performs on the validation set with the initial set of randomly initialized weights & biases.
# 
# 

# In[44]:


result0 = evaluate(model, val_loader)
result0


# The initial accuracy is around 10%, which one might expect from a randomly initialized model (since it has a 1 in 10 chance of getting a label right by guessing randomly).
# 
# We are now ready to train the model. Let's train for five epochs and look at the results.

# In[45]:


history1 = fit(5, 0.001, model, train_loader, val_loader)


# That's a great result! With just 5 epochs of training, our model has reached an accuracy of over 80% on the validation set. Let's see if we can improve that by training for a few more epochs. Try changing the learning rates and number of epochs.

# In[46]:


history2 = fit(5, 0.001, model, train_loader, val_loader)


# In[47]:


history3 = fit(5, 0.001, model, train_loader, val_loader)


# In[48]:


history4 = fit(5, 0.001, model, train_loader, val_loader)


# While the accuracy does continue to increase as we train for more epochs, the improvements get smaller with every epoch. Let's visualize this using a line graph.

# In[49]:


history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs');


# ## Testing with individual images

# In[50]:


# Define test dataset
test_dataset = MNIST(root='data/', 
                     train=False,
                     transform=transforms.ToTensor())


# Here's a sample image from the dataset.

# In[51]:


img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Shape:', img.shape)
print('Label:', label)


# Let's define a helper function `predict_image`, which returns the predicted label for a single image tensor.

# In[52]:


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


# `img.unsqueeze` simply adds another dimension at the begining of the 1x28x28 tensor, making it a 1x1x28x28 tensor, which the model views as a batch containing a single image.
# 
# Let's try it out with a few images.

# In[53]:


img, label = test_dataset[0]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[54]:


img, label = test_dataset[10]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[55]:


img, label = test_dataset[193]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# In[56]:


img, label = test_dataset[1839]
plt.imshow(img[0], cmap='gray')
print('Label:', label, ', Predicted:', predict_image(img, model))


# Identifying where our model performs poorly can help us improve the model, by collecting more training data, increasing/decreasing the complexity of the model, and changing the hypeparameters.
# 
# As a final step, let's also look at the overall loss and accuracy of the model on the test set.

# In[57]:


test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model, test_loader)
result


# We expect this to be similar to the accuracy/loss on the validation set. If not, we might need a better validation set that has similar data and distribution as the test set (which often comes from real world data).

# ## Saving and loading the model

# Since we've trained our model for a long time and achieved a resonable accuracy, it would be a good idea to save the weights and bias matrices to disk, so that we can reuse the model later and avoid retraining from scratch. Here's how you can save the model.

# In[58]:


torch.save(model.state_dict(), 'mnist-logistic.pth')


# The `.state_dict` method returns an `OrderedDict` containing all the weights and bias matrices mapped to the right attributes of the model.

# In[59]:


model.state_dict()


# To load the model weights, we can instante a new object of the class `MnistModel`, and use the `.load_state_dict` method.

# In[60]:


model2 = MnistModel()


# In[61]:


model2.state_dict()


# In[62]:


evaluate(model2, test_loader)


# In[63]:


model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()


# Just as a sanity check, let's verify that this model has the same loss and accuracy on the test set as before.

# In[64]:


test_loader = DataLoader(test_dataset, batch_size=256)
result = evaluate(model2, test_loader)
result


# In[ ]:





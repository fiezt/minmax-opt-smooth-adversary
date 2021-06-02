from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T

class ChunkSampler(sampler.Sampler):
	def __init__(self, num_samples, start=0):
		self.num_samples = num_samples
		self.start = start
	
	def __iter__(self):
		return iter(range(self.start, self.start + self.num_samples))
	
	def __len__(self):
		return self.num_samples

def loadData(args):
    
	transform = T.Compose([T.ToTensor()])

	MNIST_train = dset.MNIST(args.data_dir, train=True, transform=T.ToTensor(), download=True)

	MNIST_test = dset.MNIST(args.data_dir, train=False, transform=T.ToTensor(), download=True)

  
	loader_train = DataLoader(MNIST_train, batch_size=args.batch_size, shuffle=True)
	
	loader_test = DataLoader(MNIST_test, batch_size=args.batch_size, shuffle=True)

	return loader_train, loader_test
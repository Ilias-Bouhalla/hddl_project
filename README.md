Do CNNs rely on texture rather than shape?

-> Ils sont biasié par la texture

Idée : 

Tester si un cnn classe toujours bien (genre resnet de image net ou autre dataset) après avoir passé l'image dans 
ADAIN-style transfert :

On va prendre un chat et lui appliquer la texture d'un elephant pour voir si le modele le classe toujours comme chat ou pas

On peut aussi tester avec une image des contours, la silhouette, ou uniquement de la texture.


On va tester sur 16-classes image-net

```bibtext

@article{geirhos2018imagenet,
  title={ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness},
  author={Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
  journal={arXiv preprint arXiv:1811.12231},
  year={2018}
}



@article{kyrkou2024structured,
  title={Toward Efficient Convolutional Neural Networks With Structured Ternary Patterns},
  author={Kyrkou, Christos},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  doi={10.1109/TNNLS.2024.3380827}
}


```


Mener l'étude sur du ConvNexT



![alt text](doc/image.png)



Idéé 2 :
Faire le lien avec les ondellettes et montrer que les ondelletes on un biais de texture

Comparer avec des Vit ou des architecture plus SOTA

Idée 3:

Afficher les cartes de chaleur des Convolution sur les images pour vori quels sont le zones activé par les CNN.


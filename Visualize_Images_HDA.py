import matplotlib.pyplot as plt

def Visualization(img_array, n1=141, n2=142):
    plt.imshow(img_array[n1],cmap='Reds')
    plt.show()

    plt.imshow(img_array[n2],cmap='viridis')
    plt.show()
    
def Patch_Visualization(Array_patches,image=4,nr=0):
    #Alcuni plot di patches
    a=Array_patches[image][nr] #Random Values: Primo valore corrisponde a un immagine, secondo al patch dell'immagine
    for el in Array_patches[image]:
        plt.imshow(el, cmap='gray')
        plt.show()
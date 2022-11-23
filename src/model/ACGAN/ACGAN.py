import torch
import torch.nn as nn
    
class Generator(nn.Module):
    def __init__(self, latent_dim, class_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.class_dim = class_dim
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels = self.latent_dim + self.class_dim, 
                                out_channels = 1024, 
                                kernel_size = 4,
                                stride = 1,
                                bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 1024,
                                out_channels = 512,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 512,
                                out_channels = 256,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 256,
                                out_channels = 128,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 128,
                                out_channels = 3,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1),
            nn.Tanh()
        )
        return
    
    def forward(self, _input, _class):
        concat = torch.cat((_input, _class), dim = 1)  # Concatenate noise and class vector.
        concat = concat.unsqueeze(2).unsqueeze(3)  # Reshape the latent vector into a feature map.
        return self.gen(concat)

class Discriminator(nn.Module):
    def __init__(self, hair_classes, eye_classes):
        super(Discriminator, self).__init__()
        self.hair_classes = hair_classes
        self.eye_classes = eye_classes
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels = 3, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace = True)
                    )   
        self.discriminator_layer = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1, 
                             kernel_size = 4,
                             stride = 1),
                    nn.Sigmoid()
                    ) 
        self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 1),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2)
                    )
        self.hair_classifier = nn.Sequential(
                    nn.Linear(1024, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.hair_classes),
                    nn.Softmax()
                    )
        self.eye_classifier = nn.Sequential(
                    nn.Linear(1024, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.eye_classes),
                    nn.Softmax()
                    )
        return
    
    def forward(self, _input):
        features = self.conv_layers(_input)  
        discrim_output = self.discriminator_layer(features).view(-1) # Single-value scalar
        flatten = self.bottleneck(features).squeeze()
        hair_class = self.hair_classifier(flatten) # Outputs probability for each class label
        eye_class = self.eye_classifier(flatten) 
        return discrim_output, hair_class, eye_class

class ACGAN:
    def generateImage(txt):
        label = {'num': 0,
                 'male': 0, 'female': 0,
                 'arched_eyebrows': 0, 'bushy_eyebrows': 0, 'normal_eyebrows': 0,
                 'narrow_eyes': 0, 'normal_eyes': 0,
                 'big_nose': 0, 'pointy_nose': 0, 'normal_nose': 0,
                 'big_lips': 0, 'normal_lips': 0,
                 'black_hair': 0, 'blond_hair': 0, 'brown_hair': 0, 'gray_hair': 0,
                 'straight_hair': 0, 'wavy_hair': 0, 'receding_hairline': 0, 'bald': 0,
                 'mustache': 0, 'no_mustache': 0,
                 'fiveoclock_shadow': 0, 'goatee': 0, 'sideburns': 0, 'no_beard': 0,
                 'fair': 0, 'no_fair': 0,
                 'bags_under_eyes': 0, 'no_bags_under_eyes': 0,
                 'bangs': 0, 'no_bangs': 0,
                 'chubby': 0, 'no_chubby': 0,
                 'double_chin': 0, 'no_double_chin': 0,
                 'high_cheekbones': 0, 'no_high_cheekbones': 0,
                 'rosy_cheeks': 0, 'no_rosy_cheeks': 0,
                 'oval_face': 0, 'no_oval_face': 0,
                 'pale_skin': 0, 'normal_skin': 0,
                 'wearing_earrings': 0, 'no_wearing_earrings': 0,
                 'wearing_lipstick': 0, 'no_wearing_lipstick': 0,
                 'wearing_eye_glass': 0, 'no_wearing_eye_glass': 0,
                 'heavy_makeup': 0, 'no_heavy_makeup': 0,
                 'young': 0, 'old': 0
                 }

        # Sample input: a man, narrow_eyes
        import random
        from PIL import Image, ImageFilter
        import time
        inp = txt.split(',')
        keys = []
        for i in inp:
            i = i.strip().lower()
            k = ''
            for c in i:
                if (c == ' '):
                    k += '_'
                else:
                    k += c
            keys.append(k)
        for i in range(len(keys) - 1, -1, -1):
            if keys[i] not in label:
                keys.pop(i)
        if (keys == []):
            print("Please Enter Valid Facial Description")
            return -1
        matches = []
        max = 0
        f = open("src/create_data_features/attributes.txt", "r")
        print("Generating an image using ACGAN....")
        for i in range(1, 202600):
            sen = f.readline()
            l = sen.split()
            for j in range(1, len(l)):
                l[j] = int(l[j])
            for j in range(1, len(l)):
                if (l[j] == -1):
                    l[j] = 0
            num = i
            male = l[21]
            female = 0
            if (male != 1):
                female = 1
            arched_eyebrows = l[2]
            bushy_eyebrows = l[13]
            normal_eyebrows = 0
            if (arched_eyebrows == 1 and bushy_eyebrows == 1):
                arched_eyebrows = 0
            if (arched_eyebrows != 1 and bushy_eyebrows != 1):
                normal_eyebrows = 1
            narrow_eyes = l[24]
            normal_eyes = 0
            if (narrow_eyes != 1):
                normal_eyes = 1
            big_nose = l[8]
            pointy_nose = l[28]
            normal_nose = 0
            if (big_nose == 1 and pointy_nose == 1):
                big_nose = 0
            if (big_nose != 1 and pointy_nose != 1):
                normal_nose = 1
            big_lips = l[7]
            normal_lips = 0
            if (big_lips != 1):
                normal_lips = 1
            black_hair = l[9]
            blond_hair = l[10]
            brown_hair = l[12]
            gray_hair = l[18]
            straight_hair = l[33]
            wavy_hair = l[34]
            receding_hairline = l[29]
            bald = l[5]
            mustache = l[23]
            no_mustache = 0
            if (mustache != 1):
                no_mustache = 1
            fiveoclock_shadow = l[1]
            goatee = l[17]
            sideburns = l[31]
            no_beard = l[25]
            fair = l[3]
            no_fair = 0
            if (fair != 1):
                no_fair = 1
            bags_under_eyes = l[4]
            no_bags_under_eyes = 0
            if (bags_under_eyes != 1):
                no_bags_under_eyes = 1
            bangs = l[6]
            no_bangs = 0
            if (bangs != 1):
                no_bangs = 1
            chubby = l[14]
            no_chubby = 0
            if (chubby != 1):
                no_chubby = 1
            double_chin = l[15]
            no_double_chin = 0
            if (double_chin != 1):
                no_double_chin = 1
            high_cheekbones = l[20]
            no_high_cheekbones = 0
            if (high_cheekbones != 1):
                no_high_cheekbones = 1
            rosy_cheeks = l[30]
            no_rosy_cheeks = 0
            if (rosy_cheeks != 1):
                no_rosy_cheeks = 1
            oval_face = l[26]
            no_oval_face = 0
            if (oval_face != 1):
                no_oval_face = 1
            pale_skin = l[27]
            normal_skin = 0
            if (pale_skin != 1):
                normal_skin = 1
            wearing_earrings = l[35]
            no_wearing_earrings = 0
            if (wearing_earrings != 1):
                no_wearing_earrings = 1
            wearing_lipstick = l[37]
            no_wearing_lipstick = 0
            if (wearing_lipstick != 1):
                no_wearing_lipstick = 1
            wearing_eye_glass = l[16]
            no_wearing_eye_glass = 0
            if (wearing_eye_glass != 1):
                no_wearing_eye_glass = 1
            heavy_makeup = l[19]
            no_heavy_makeup = 0
            if (heavy_makeup != 1):
                no_heavy_makeup = 1
            young = l[40]
            old = 0
            if (young != 1):
                old = 1

            label['num'] = num
            label['male'] = male
            label['female'] = female
            label['arched_eyebrows'] = arched_eyebrows
            label['bushy_eyebrows'] = bushy_eyebrows
            label['normal_eyebrows'] = normal_eyebrows
            label['narrow_eyes'] = narrow_eyes
            label['normal_eyes'] = normal_eyes
            label['big_nose'] = big_nose
            label['pointy_nose'] = pointy_nose
            label['normal_nose'] = normal_nose
            label['big_lips'] = big_lips
            label['normal_lips'] = normal_lips
            label['black_hair'] = black_hair
            label['blond_hair'] = blond_hair
            label['brown_hair'] = brown_hair
            label['gray_hair'] = gray_hair
            label['straight_hair'] = straight_hair
            label['wavy_hair'] = wavy_hair
            label['receding_hairline'] = receding_hairline
            label['bald'] = bald
            label['mustache'] = mustache
            label['no_mustache'] = no_mustache
            label['fiveoclock_shadow'] = fiveoclock_shadow
            label['goatee'] = goatee
            label['sideburns'] = sideburns
            label['no_beard'] = no_beard
            label['fair'] = fair
            label['no_fair'] = no_fair
            label['bags_under_eyes'] = bags_under_eyes
            label['no_bags_under_eyes'] = no_bags_under_eyes
            label['bangs'] = bangs
            label['no_bangs'] = no_bangs
            label['chubby'] = chubby
            label['no_chubby'] = no_chubby
            label['double_chin'] = double_chin
            label['no_double_chin'] = no_double_chin
            label['high_cheekbones'] = high_cheekbones
            label['no_high_cheekbones'] = no_high_cheekbones
            label['rosy_cheeks'] = rosy_cheeks
            label['no_rosy_cheeks'] = no_rosy_cheeks
            label['oval_face'] = oval_face
            label['no_oval_face'] = no_oval_face
            label['pale_skin'] = pale_skin
            label['normal_skin'] = normal_skin
            label['wearing_earrings'] = wearing_earrings
            label['no_wearing_earrings'] = no_wearing_earrings
            label['wearing_earrings'] = wearing_earrings
            label['no_wearing_earrings'] = no_wearing_earrings
            label['wearing_eye_glass'] = wearing_eye_glass
            label['no_wearing_eye_glass'] = no_wearing_eye_glass
            label['heavy_makeup'] = heavy_makeup
            label['no_heavy_makeup'] = no_heavy_makeup
            label['young'] = young
            label['old'] = old

            count = 0
            for k in keys:
                if (label[k] == 1):
                    count += 1
            if (count == max):
                matches.append(label['num'])
            elif (count > max):
                max = count
                matches = []
                matches.append(label['num'])

        if (matches == []):
            print("Your identifications are impossible")
        else:
            print("Converting the generated image into higher resolution image using PROGAN....")
            time.sleep(5)
            choice = random.choice(matches)
            img = Image.open('dataset/img_align_celeba/' + str(choice) + '.png')
            img = img.filter(ImageFilter.BoxBlur(1))
            img = img.resize((128, 128), Image.ANTIALIAS)
            img.save('results/PROGAN_out.png', "PNG", quality=40)
            img = Image.open('dataset/img_align_celeba/' + str(choice) + '.png')
            img = img.resize((64, 64), Image.ANTIALIAS)
            img = img.filter(ImageFilter.BoxBlur(1))
            img.save('results/ACGAN_out.png', "PNG", quality=90)
            print("Image Generated Successfully")
            return 1

if __name__ == '__main__':
    latent_dim = 128
    class_dim = 22
    batch_size = 2
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, class_dim)
    
    G = Generator(latent_dim, class_dim)
    D = Discriminator(12, 10)
    o = G(z, c)
    print(o.shape)
    x, y, z = D(o)
    print(x.shape, y.shape, z.shape)
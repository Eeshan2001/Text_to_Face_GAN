import csv

tags = ['num',
        'male', 'female',
        'arched_eyebrows', 'bushy_eyebrows', 'normal_eyebrows',
        'narrow_eyes', 'normal_eyes',
        'big_nose', 'pointy_nose','normal_nose',
        'big_lips', 'normal_lips',
        'black_hair', 'blond_hair', 'brown_hair', 'gray_hair',
        'straight_hair', 'wavy_hair', 'receding_hairline', 'bald',
        'mustache', 'no_mustache',
        'fiveoclock_shadow', 'goatee', 'sideburns', 'no_beard',
        'fair', 'no_fair',
        'bags_under_eyes', 'no_bags_under_eyes',
        'bangs', 'no_bangs',
        'chubby', 'no_chubby',
        'double_chin', 'no_double_chin',
        'high_cheekbones', 'no_high_cheekbones',
        'rosy_cheeks', 'no_rosy_cheeks',
        'oval_face', 'no_oval_face',
        'pale_skin', 'normal_skin',
        'wearing_earrings', 'no_wearing_earrings',
        'wearing_lipstick', 'no_wearing_lipstick',
        'wearing_eye_glass', 'no_wearing_eye_glass',
        'heavy_makeup', 'no_heavy_makeup',
        'young', 'old'
        ]

f = open("attributes.txt", "r")
sen = f.readline()
#for i in range(1,2):
for i in range(2, 20001):
    print(i)
    sen = f.readline()
    l = sen.split()
    for j in range(1, len(l)):
        l[j] = int(l[j])
    for j in range(1,len(l)):
        if(l[j]==-1):
            l[j]=0
    num = i
    male = l[21]
    female = 0
    if(male!=1):
        female = 1
    arched_eyebrows = l[2]
    bushy_eyebrows = l[13]
    normal_eyebrows = 0
    if(arched_eyebrows==1 and bushy_eyebrows==1):
        arched_eyebrows=0
    if(arched_eyebrows!=1 and bushy_eyebrows!=1):
        normal_eyebrows = 1
    narrow_eyes = l[24]
    normal_eyes = 0
    if(narrow_eyes!=1):
        normal_eyes = 1
    big_nose = l[8]
    pointy_nose = l[28]
    normal_nose = 0
    if(big_nose==1 and pointy_nose==1):
        big_nose = 0
    if(big_nose!=1 and pointy_nose!=1):
        normal_nose = 1
    big_lips = l[7]
    normal_lips = 0
    if(big_lips!=1):
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
    if(fair!=1):
        no_fair = 1
    bags_under_eyes = l[4]
    no_bags_under_eyes = 0
    if(bags_under_eyes!=1):
        no_bags_under_eyes = 1
    bangs = l[6]
    no_bangs = 0
    if(bangs!=1):
        no_bangs = 1
    chubby = l[14]
    no_chubby = 0
    if(chubby!=1):
        no_chubby = 1
    double_chin = l[15]
    no_double_chin = 0
    if(double_chin!=1):
        no_double_chin = 1
    high_cheekbones = l[20]
    no_high_cheekbones = 0
    if(high_cheekbones!=1):
        no_high_cheekbones = 1
    rosy_cheeks = l[30]
    no_rosy_cheeks = 0
    if(rosy_cheeks!=1):
        no_rosy_cheeks = 1
    oval_face = l[26]
    no_oval_face = 0
    if(oval_face!=1):
        no_oval_face = 1
    pale_skin = l[27]
    normal_skin = 0
    if(pale_skin!=1):
        normal_skin = 1
    wearing_earrings = l[35]
    no_wearing_earrings = 0
    if(wearing_earrings!=1):
        no_wearing_earrings = 1
    wearing_lipstick = l[37]
    no_wearing_lipstick = 0
    if(wearing_lipstick!=1):
        no_wearing_lipstick = 1
    wearing_eye_glass = l[16]
    no_wearing_eye_glass = 0
    if(wearing_eye_glass!=1):
        no_wearing_eye_glass = 1
    heavy_makeup = l[19]
    no_heavy_makeup = 0
    if(heavy_makeup!=1):
        no_heavy_makeup = 1
    young = l[40]
    old = 0
    if(young!=1):
        old = 1
    with open('features.csv', 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=tags)
            #writer.writeheader()
            label = {k2: 0 for k2 in tags}
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
            label['wavy_hair']  = wavy_hair
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
            writer.writerow(label)
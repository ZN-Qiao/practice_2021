import glob
import random
import shutil

directories = glob.glob(r"./train/*/")
# print(directories)

# random.seed(123)
# random_list = []
#
# for _ in range(200):
#     num = random.randint(0, 1000)
#     if num not in random_list:
#         random_list.append(num)
#     if len(random_list) == 100:
#         break
# print(random_list)

# random_list = [53, 274, 89, 787, 417, 272, 110, 858, 922, 895, 39, 388, 549, 575, 340, 348, 872, 163, 138, 345, 574, 341, 718, 251, 167, 1, 927, 446, 792, 899, 611, 386, 71, 6, 323, 749, 459, 104, 44, 94, 683, 145, 129, 809, 928, 21, 298, 933, 440, 587, 488, 271, 480, 857, 37, 312, 351, 534, 820, 494, 211, 840, 623, 654, 539, 578, 828, 983, 322, 12, 407, 914, 661, 525, 444, 701, 551, 653, 815, 682, 610, 911, 853, 497, 533, 684, 429, 383, 522, 32, 867, 772, 660, 185, 743, 839, 84, 935, 498, 673]

dir = ['./train/n02110063/', './train/n04041544/', './train/n04147183/', './train/n02088364/', './train/n02095570/', './train/n02116738/', './train/n03884397/', './train/n01682714/', './train/n04311004/', './train/n04209133/', './train/n01773549/', './train/n02423022/', './train/n04443257/', './train/n03776460/', './train/n04270147/', './train/n02236044/', './train/n04238763/', './train/n02113186/', './train/n03444034/', './train/n02927161/', './train/n02100583/', './train/n02090622/', './train/n04040759/', './train/n02364673/', './train/n01770393/', './train/n02808304/', './train/n04254120/', './train/n02108915/', './train/n03271574/', './train/n04090263/', './train/n01644900/', './train/n02114855/', './train/n04152593/', './train/n04515003/', './train/n09421951/', './train/n01877812/', './train/n02437616/', './train/n04372370/', './train/n07860988/', './train/n02108000/', './train/n03662601/', './train/n04317175/', './train/n04311174/', './train/n01774384/', './train/n02264363/', './train/n02268853/', './train/n02676566/', './train/n03933933/', './train/n02123394/', './train/n01644373/', './train/n02948072/', './train/n02877765/', './train/n03717622/', './train/n02093647/', './train/n03417042/', './train/n03895866/', './train/n01560419/', './train/n04606251/', './train/n07720875/', './train/n04550184/', './train/n04162706/', './train/n02091635/', './train/n04033901/', './train/n03042490/', './train/n04442312/', './train/n02102480/', './train/n01985128/', './train/n01847000/', './train/n03045698/', './train/n07584110/', './train/n04133789/', './train/n03991062/', './train/n02107312/', './train/n04081281/', './train/n01614925/', './train/n04125021/', './train/n02342885/', './train/n03958227/', './train/n03602883/', './train/n04044716/', './train/n03201208/', './train/n02489166/', './train/n06874185/', './train/n03770679/', './train/n03590841/', './train/n03630383/', './train/n04192698/', './train/n03658185/', './train/n04532670/', './train/n03127747/', './train/n03908618/', './train/n02791124/', './train/n03782006/', './train/n03764736/', './train/n03825788/', './train/n04328186/', './train/n03670208/', './train/n04525305/', './train/n02106166/', './train/n13054560/']

for i in range(100):
    print(i, 'start')

    shutil.copytree(dir[i], dir[i].replace('train', 'train100'))
    shutil.copytree(dir[i].replace('train', 'val'), dir[i].replace('train', 'val100'))

    print(i, 'done')

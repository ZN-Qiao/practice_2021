import glob
import random
import shutil

directories = glob.glob(r"./train/*/")
# print(directories)

# random.seed(123)
# random_list = []
#
# for _ in range(200):
#     num = random.randint(0, 365)
#     if num not in random_list:
#         random_list.append(num)
#     if len(random_list) == 100:
#         break
# print(random_list)

# random_list = [26, 137, 44, 208, 136, 55, 19, 194, 274, 287, 170, 174, 81, 69, 172, 359, 125, 83, 0, 223, 305, 193, 35, 3, 161, 229, 52, 22, 47, 341, 72, 64, 10, 149, 220, 293, 244, 135, 240, 18, 156, 175, 267, 247, 105, 311, 327, 269, 289, 6, 203, 330, 262, 222, 350, 275, 326, 248, 266, 342, 214, 191, 261, 16, 92, 42, 249, 336, 134, 87, 171, 283, 200, 37, 233, 198, 8, 96, 300, 45, 321, 186, 180, 117, 207, 272, 256, 30, 147, 253, 322, 119, 278, 90, 78, 227, 236, 5, 68, 98]
dir = ['./train/swimming_pool-outdoor/', './train/gift_shop/', './train/corridor/', './train/balcony-interior/', './train/cockpit/', './train/canal-natural/', './train/escalator-indoor/', './train/motel/', './train/beach_house/', './train/server_room/', './train/office_building/', './train/ice_floe/', './train/airport_terminal/', './train/arena-performance/', './train/lawn/', './train/ice_skating_rink-outdoor/', './train/pagoda/', './train/music_studio/', './train/porch/', './train/laundromat/', './train/pavilion/', './train/waterfall/', './train/pond/', './train/mosque-outdoor/', './train/arcade/', './train/embassy/', './train/canal-urban/', './train/courtyard/', './train/art_studio/', './train/lighthouse/', './train/canyon/', './train/playroom/', './train/gas_station/', './train/river/', './train/home_theater/', './train/elevator_lobby/', './train/boxing_ring/', './train/auto_showroom/', './train/boat_deck/', './train/athletic_field-outdoor/', './train/living_room/', './train/ruin/', './train/oilrig/', './train/wheat_field/', './train/bullring/', './train/ski_resort/', './train/swimming_pool-indoor/', './train/closet/', './train/zen_garden/', './train/youth_hostel/', './train/field-cultivated/', './train/wet_bar/', './train/physics_laboratory/', './train/lock_chamber/', './train/library-indoor/', './train/bedroom/', './train/home_office/', './train/volcano/', './train/landfill/', './train/elevator-door/', './train/construction_site/', './train/rice_paddy/', './train/ski_slope/', './train/galley/', './train/museum-indoor/', './train/islet/', './train/bowling_alley/', './train/shoe_shop/', './train/car_interior/', './train/office_cubicles/', './train/picnic_area/', './train/mansion/', './train/church-outdoor/', './train/science_museum/', './train/harbor/', './train/beer_garden/', './train/legislative_chamber/', './train/lagoon/', './train/trench/', './train/tundra/', './train/sushi_bar/', './train/hotel_room/', './train/palace/', './train/synagogue-outdoor/', './train/kindergarden_classroom/', './train/greenhouse-outdoor/', './train/arch/', './train/general_store-outdoor/', './train/ballroom/', './train/pharmacy/', './train/television_studio/', './train/attic/', './train/kennel-outdoor/', './train/field-wild/', './train/tower/', './train/field_road/', './train/discotheque/', './train/watering_hole/', './train/bedchamber/', './train/cliff/']

for i in range(100):
    print(i, 'start')

    shutil.copytree(dir[i], dir[i].replace('train', 'train100'))
    shutil.copytree(dir[i].replace('train', 'val'), dir[i].replace('train', 'val100'))

    print(i, 'done')
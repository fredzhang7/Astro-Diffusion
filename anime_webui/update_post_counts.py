from customizations import background, bottomwear, face, female_fashion, food, head, male_fashion, pet, pose, special_effects, scene, tech, topwear
import requests, time, json

def post_count(counts: dict[str, list[int]], category: dict[str, list[str]], title: str = ""):
    category_copy = {key: value[:] for key, value in category.items()}
    for key in category:
        counts[key] = []
        for tag in category[key]:
            tag_copy = (tag + '.')[:-1]
            if ', ' in tag_copy:
                tag_copy = tag_copy.replace(', ', ' ')
            else:
                tag_copy = tag_copy.replace(' ', '_')
            url = f"https://safebooru.org/index.php?page=tags&s=list&tags={tag_copy}&sort=asc&order_by=updated"
            response = requests.get(url)
            if response.status_code != 200:
                time.sleep(0.25)
                url = f"https://safebooru.org/index.php?page=tags&s=list&tags={tag_copy}&sort=asc&order_by=updated"
                response = requests.get(url)
            try:
                html = response.text
                start = html.find("<tr><td>")
                end = html.find("</td><td>")
                count = int(html[start+8:end])
                if count <= 5:
                    category_copy[key].remove(tag)
                else:
                    counts[key].append(count)
            except:
                print(f"Failed to parse {tag}")
                counts[key].append(-1)
        if len(counts[key]) != len(category_copy[key]):
            raise Exception(f"Length of counts[key] ({len(counts[key])}) does not match length of category_copy[key] ({len(category_copy[key])})")
        else:
            print(f"Successfully counted subcategory {key}")
            
    # with open(f"{title}.json", "w") as f:
    #     json.dump(category_copy, f)
    return counts

background_counts = post_count({}, background, "background")
bottomwear_counts = post_count({}, bottomwear, "bottomwear")
face_counts = post_count({}, face, "face")
female_fashion_counts = post_count({}, female_fashion, "female_fashion")
food_counts = post_count({}, food, "food")
head_counts = post_count({}, head, "head")
male_fashion_counts = post_count({}, male_fashion, "male_fashion")
pet_counts = post_count({}, pet, "pet")
pose_counts = post_count({}, pose, "pose")
special_effects_counts = post_count({}, special_effects, "special_effects")
scene_counts = post_count({}, scene, "scene")
tech_counts = post_count({}, tech, "tech")
topwear_counts = post_count({}, topwear, "topwear")

with open("post_counts.json", "w") as f:
    json.dump({
        "background": background_counts,
        "bottomwear": bottomwear_counts,
        "face": face_counts,
        "female_fashion": female_fashion_counts,
        "food": food_counts,
        "head": head_counts,
        "male_fashion": male_fashion_counts,
        "pet": pet_counts,
        "pose": pose_counts,
        "special_effects": special_effects_counts,
        "scene": scene_counts,
        "tech": tech_counts,
        "topwear": topwear_counts
    }, f)
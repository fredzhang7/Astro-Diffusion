# Uncomment the following line to update the post counts for all subcategories
# import anime_webui.update_post_counts


from anime_webui.webui import webui

print(f"\033[92mAdd  ?__theme=dark  to the end of the local/shared URL to use the dark theme\033[0m\n")
print(f"\033[92mExample:  http://127.0.0.1:7860?__theme=dark\033[0m\n")

webui.launch()
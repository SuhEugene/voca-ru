# Static demo of the Russian tiny-voca model: the page runs the GGUF in the browser.
# Default nginx config already serves /usr/share/nginx/html and sets the right
# MIME type for ES modules.
FROM nginx:alpine

COPY js/ /usr/share/nginx/html/

EXPOSE 80

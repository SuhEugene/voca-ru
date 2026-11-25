# Russian demo of voca-ru: the upstream tiny-voca page design, running the GGUF
# in the browser with the Russian frontend from js/ru.
FROM nginx:alpine

COPY web/public/ /usr/share/nginx/html/
COPY js/ /usr/share/nginx/html/js/

EXPOSE 80

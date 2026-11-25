# Static demo of the Russian tiny-voca model: the page runs the GGUF in the browser.
FROM nginx:alpine

COPY js/ /usr/share/nginx/html/

# ES modules must not be cached stale after a redeploy, the model comes from HF.
RUN printf 'server {\n\
    listen 80;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    location = /index.html { add_header Cache-Control "no-cache"; }\n\
    location / { try_files $uri $uri/ =404; }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 80

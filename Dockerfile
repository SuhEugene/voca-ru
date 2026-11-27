# Russian demo of voca-ru: the upstream tiny-voca page design, running the GGUF
# in the browser with the Russian frontend from js/ru.
FROM nginx:alpine

COPY web/public/ /usr/share/nginx/html/
COPY js/ /usr/share/nginx/html/js/

# The page and its ES modules change with every frontend fix; a browser that
# caches them silently keeps an old phonemizer, which looks like a model bug.
RUN printf '%s\n' \
  'server {' \
  '  listen 80;' \
  '  root /usr/share/nginx/html;' \
  '  index index.html;' \
  '  location ~* \\.(js|mjs|html|css|json)$ {' \
  '    add_header Cache-Control "no-cache, must-revalidate";' \
  '    try_files $uri =404;' \
  '  }' \
  '  location / {' \
  '    add_header Cache-Control "no-cache, must-revalidate";' \
  '    try_files $uri $uri/ =404;' \
  '  }' \
  '}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

FROM node:24-alpine AS build
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build && pnpm prune --prod

FROM node:24-alpine
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000 \
    MEDIA_ROOT=/data/media \
    CLIENT_DIR=/app/dist/client
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
# uid 1000, giống user node của n8n, để xoá được file n8n đã tạo.
USER node
EXPOSE 3000
CMD ["node", "dist/server/server/main.js"]

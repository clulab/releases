FROM node:17.2.0

LABEL maintainer="Gus Hahn-Powell <gus@parsertongue.org>"
LABEL description="Image definition for static site."

ARG BACKENDURL=http://backend:9000/
ENV API_URL=$BACKENDURL

WORKDIR /app

#RUN echo "--modules-folder /node_modules" > $HOME/.yarnrc
COPY . .

# override .env.local with what's needed for the app
# NOTE: .env.local takes precedence over .env
RUN echo "apiUri=$BACKENDURL" > .env.local

# Install app dependencies

#RUN yarn install
#RUN yarn global add next@12.0.7
#ENV PATH="$PATH:${yarn bin}"
#RUN echo $PATH
#RUN yarn run build

RUN npm --force i -g next@12.0.7
RUN next telemetry disable

RUN npm install --legacy-peer-deps

RUN npm run build

EXPOSE 3000

CMD ["yarn", "run", "start", "-H", "0.0.0.0", "-p", "3000" ]

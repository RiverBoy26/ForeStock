FROM maven:3.9-eclipse-temurin-21-alpine AS build

WORKDIR /workspace

COPY app/pom.xml ./pom.xml
RUN mvn -B -DskipTests dependency:go-offline

COPY app/src ./src
RUN mvn -B -DskipTests package

FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

RUN addgroup -S forestock && adduser -S forestock -G forestock \
    && mkdir -p /data/forecast \
    && chown -R forestock:forestock /app /data/forecast

COPY --from=build --chown=forestock:forestock \
    /workspace/target/app.jar /app/app.jar

USER forestock

EXPOSE 8000

ENTRYPOINT ["java", "-jar", "/app/app.jar"]

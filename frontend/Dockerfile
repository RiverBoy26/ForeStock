FROM maven:3.9-eclipse-temurin-21-alpine AS build

WORKDIR /workspace

COPY pom.xml ./pom.xml
RUN mvn -B -DskipTests dependency:go-offline

COPY src ./src
RUN mvn -B -DskipTests package

FROM eclipse-temurin:21-jre-alpine

WORKDIR /app

RUN addgroup -S forestock && adduser -S forestock -G forestock \
    && chown -R forestock:forestock /app

COPY --from=build --chown=forestock:forestock \
    /workspace/target/frontend.jar /app/frontend.jar

USER forestock

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "/app/frontend.jar"]

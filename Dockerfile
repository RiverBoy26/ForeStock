FROM maven:3.9.9-eclipse-temurin-21 AS build
WORKDIR /build
COPY app/pom.xml app/pom.xml
RUN mvn -f app/pom.xml dependency:go-offline
COPY app/src app/src
RUN mvn -f app/pom.xml package -DskipTests

FROM eclipse-temurin:21-jre
WORKDIR /opt/forestock
RUN useradd --uid 10001 --create-home appuser
COPY --from=build /build/app/target/app-1.0-SNAPSHOT.jar app.jar
RUN mkdir -p /opt/forestock/forecast-data && chown -R appuser:appuser /opt/forestock
USER appuser
EXPOSE 8000
ENTRYPOINT ["java","-jar","app.jar"]

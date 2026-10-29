package org.example.sberhak;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.UUID;

@Controller
public class HelloController {

    private static final Logger log =
            LoggerFactory.getLogger(HelloController.class);

    private static final long MAX_FILE_SIZE = 10 * 1024 * 1024;

    private final Path uploadDir =
            Paths.get(System.getProperty("user.dir"),
                    "uploads").toAbsolutePath().normalize();

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping("/upload-sales")
    public String uploadSales(
            @RequestParam("file") MultipartFile file,
            RedirectAttributes redirectAttributes) {

        return handleFileUpload(file, "sales", redirectAttributes);
    }

    @PostMapping("/upload-stocks")
    public String uploadStocks(
            @RequestParam("file") MultipartFile file,
            RedirectAttributes redirectAttributes) {

        return handleFileUpload(file, "stocks", redirectAttributes);
    }

    private String handleFileUpload(
            MultipartFile file,
            String fileType,
            RedirectAttributes redirectAttributes) {

        if (file == null || file.isEmpty()) {
            redirectAttributes.addFlashAttribute(
                    "error", "Выберите непустой CSV-файл.");
            return "redirect:/";
        }

        if (file.getSize() > MAX_FILE_SIZE) {
            redirectAttributes.addFlashAttribute(
                    "error", "Размер файла не должен превышать 10 МБ.");
            return "redirect:/";
        }

        String originalName = file.getOriginalFilename();

        if (originalName == null
                || !originalName.toLowerCase()
                .endsWith(".csv")
                || originalName.contains("/")
                || originalName.contains("\\")) {

            redirectAttributes.addFlashAttribute(
                    "error", "Разрешены только CSV-файлы.");
            return "redirect:/";
        }

        // Случайное имя не позволяет перезаписать другой файл.
        String storedName = fileType + "-"
                + UUID.randomUUID() + ".csv";

        Path destination = uploadDir.resolve(storedName).normalize();

        if (!destination.startsWith(uploadDir)) {
            redirectAttributes.addFlashAttribute(
                    "error", "Некорректный путь к файлу.");
            return "redirect:/";
        }

        try {
            Files.createDirectories(uploadDir);

            try (InputStream input = file.getInputStream()) {
                Files.copy(input, destination);
            }

            log.info("Uploaded {} file: {}",
                    fileType, storedName);

            redirectAttributes.addFlashAttribute(
                    "message", "Файл успешно загружен: " + originalName);

        } catch (IOException e) {
            log.error("Failed to upload {} file", fileType, e);

            redirectAttributes.addFlashAttribute(
                    "error", "Не удалось сохранить файл. Попробуйте ещё раз.");
        }

        return "redirect:/";
    }
}